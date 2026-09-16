"""
Simple email/password account system, stored in Firestore.

This does NOT use Firebase's separate "Authentication" product -- that's
built for apps with a JavaScript frontend, which Streamlit isn't. Instead,
accounts are plain Firestore documents (collection "accounts", one document
per email) holding a bcrypt password hash. bcrypt is a one-way hash built
specifically for passwords: even with full database access, the original
password cannot be recovered from it.

Each account document:
    {
        "password_hash": bytes,
        "free_used": int,                          # free-tier searches used in the current 24h window
        "free_window_start": datetime | None,       # when that window started; None = never used yet
        "subscribed_until": datetime | None,       # manually granted paid access
        "subscription_search_limit": int,          # searches allowed for the CURRENT paid period
        "subscription_used": int,                  # searches used in the current paid period
        "plan": str,                               # "normal" | "pro" | "max" -- see run_assistant.PLAN_MODELS
        "session_token": str,                      # current "remember me" token, or "" once logged out
        "session_token_created_at": datetime,      # for expiring old tokens
        "field_of_study": str,                     # optional, self-reported, blank by default
        "academic_level": str,                     # optional, one of ACADEMIC_LEVELS, blank by default
        "tone": str,                                # optional, one of TONE_OPTIONS, blank by default
        "custom_instructions": str,                 # optional free text, blank by default
        "password_hint": str,                       # optional, self-set at signup, blank by default
        "reset_token": str,                         # current password-reset token, or "" once used/cleared
        "reset_token_created_at": datetime,         # for expiring old reset tokens
    }

A "subscribed" account (see grant_subscription/is_subscribed below) gets a
separate, generous-but-finite search allowance while subscribed_until is in
the future -- like a real paid plan (e.g. ChatGPT Plus), not literally
unlimited, so no single subscriber can exhaust the whole site's budget.
This is the manual-payment model (customer pays the owner directly, e.g.
bank transfer, exactly like a resold ChatGPT Team seat; the owner then
grants access here) -- separate from, and simpler than, an automated
payment gateway. The site-wide emergency cap (global_limit.py) still
applies on top of this for every account, subscribed or not.
"""

import re
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from firebase_admin import firestore

from db import _get_client

# Free (unsubscribed) tier: a small, recurring daily allowance, not a
# one-time lifetime trial -- gives a repeatable taste of the app instead of
# a single moment that's either converted or gone forever, matching how
# real comparable products (e.g. Perplexity's free tier) actually do this.
# Paired with a cheaper model (see run_assistant.PLAN_MODELS["free"]) so
# this stays affordable to give away indefinitely, not just during a trial.
FREE_DAILY_SEARCH_LIMIT = 5
FREE_DAILY_WINDOW = timedelta(hours=24)

# "Remember me" across browser refreshes: a random opaque token is stored on
# the account and also placed in the page URL (?t=...), so app.py can
# silently restore the session on load instead of asking for the password
# again every time. Only the single most-recently-issued token is valid --
# logging in again overwrites it, so this is one "remembered" session per
# account, not a token list. Expires after SESSION_TOKEN_MAX_AGE_DAYS so a
# stale/copied link doesn't work forever.
SESSION_TOKEN_MAX_AGE_DAYS = 30

# Password reset: a random opaque token emailed as a link (?reset_token=...),
# short-lived (unlike the "remember me" token above, which is meant to last)
# since a reset link sitting in an inbox is a real, if small, exposure
# window. Same single-active-token pattern -- requesting a new reset link
# invalidates any previous one.
RESET_TOKEN_MAX_AGE = timedelta(hours=1)

# Kept as plain strings here (not imported from run_assistant.py) to keep
# auth.py free of any model-calling dependency -- it only needs to validate
# and store the plan name, not know which real models each one maps to.
PLANS = ("normal", "pro", "max")

# Self-reported profile, used to gently steer tone/vocabulary in the AI's
# own interpretive writing (never to change which real evidence is found or
# what it says) -- see app.py's profile_context_note(). Same for every
# account regardless of plan; it's just two text fields, no extra model
# cost, so there's no reason to gate it.
ACADEMIC_LEVELS = ("عام", "بكالوريوس", "ماجستير", "دكتوراه")
TONE_OPTIONS = ("افتراضي", "رسمي", "مبسّط ومباشر", "مفصّل وعميق")

CUSTOM_INSTRUCTIONS_MAX_LEN = 500  # a short note, not a second essay

# Generous but finite -- like a real paid plan's usage cap, not literally
# unlimited. Priced (see ROADMAP.md) so each tier stays profitable at
# *typical* usage; a subscriber using every single search in the period
# would cost more than normal/pro's price (a deliberate bet, backed by
# real precedent -- Perplexity's own published usage patterns -- that most
# subscribers never get near a generous cap), with global_limit.py's
# site-wide cap as the backstop if that bet is wrong. Easy to adjust
# per-plan later once real usage data exists.
SUBSCRIPTION_SEARCH_LIMITS = {
    "normal": 40,
    "pro": 30,
    "max": 20,
}

# Monthly price per plan, in IQD -- see ROADMAP.md for the pricing research
# behind these numbers. Used by wayl_client.py / app.py to create a real
# checkout link; kept here alongside the other plan metadata rather than in
# app.py so the price and the plan it belongs to never drift apart.
PLAN_PRICES_IQD = {
    "normal": 6000,
    "pro": 7000,
    "max": 8500,
}

SUBSCRIPTION_DAYS = 30  # one paid period -- see grant_subscription() below

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised for account-creation/login problems meant to be shown to the user."""


def _accounts():
    return _get_client().collection("accounts")


def create_account(email: str, password: str, password_hint: str = "") -> None:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise AuthError("البريد الإلكتروني غير صالح.")
    if len(password) < 8:
        raise AuthError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")

    doc_ref = _accounts().document(email)
    if doc_ref.get().exists:
        raise AuthError("هذا البريد الإلكتروني مسجّل بالفعل.")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    doc_ref.set({
        "password_hash": password_hash,
        "password_hint": password_hint.strip(),
    })


def verify_login(email: str, password: str) -> bool:
    email = email.strip().lower()
    if not email:
        # An empty string is not a valid Firestore document id and would
        # otherwise crash the whole app instead of just failing the login.
        return False
    doc = _accounts().document(email).get()
    if not doc.exists:
        return False
    stored_hash = doc.to_dict().get("password_hash")
    if not stored_hash:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash)


def get_account(email: str) -> dict | None:
    email = email.strip().lower()
    doc = _accounts().document(email).get()
    return doc.to_dict() if doc.exists else None


def create_session_token(email: str) -> str:
    """Issues a new "remember me" token for this account and returns it, to
    be stored in the page URL by app.py. Overwrites any previous token."""
    email = email.strip().lower()
    token = secrets.token_urlsafe(32)
    _accounts().document(email).set({
        "session_token": token,
        "session_token_created_at": datetime.now(timezone.utc),
    }, merge=True)
    return token


def verify_session_token(token: str) -> str | None:
    """Returns the account email for a still-valid "remember me" token, or
    None if it's missing, doesn't match any account, or has expired."""
    if not token:
        return None
    matches = _accounts().where("session_token", "==", token).limit(1).stream()
    for doc in matches:
        data = doc.to_dict()
        created_at = data.get("session_token_created_at")
        if created_at and datetime.now(timezone.utc) - created_at <= timedelta(days=SESSION_TOKEN_MAX_AGE_DAYS):
            return doc.id
    return None


def clear_session_token(email: str) -> None:
    """Invalidates the account's "remember me" token (e.g. on logout)."""
    email = email.strip().lower()
    _accounts().document(email).set({"session_token": ""}, merge=True)


def create_reset_token(email: str) -> str | None:
    """Issues a new password-reset token for this account and returns it
    (to be emailed as a link), or None if no account exists for this email
    -- callers should show the same neutral "if this email is registered..."
    message either way, so a forgot-password attempt can't be used to probe
    which emails have accounts. Overwrites any previous reset token."""
    email = email.strip().lower()
    if get_account(email) is None:
        return None
    token = secrets.token_urlsafe(32)
    _accounts().document(email).set({
        "reset_token": token,
        "reset_token_created_at": datetime.now(timezone.utc),
    }, merge=True)
    return token


def verify_reset_token(email: str, token: str) -> bool:
    """True if `token` is this specific account's current, still-valid
    reset token."""
    if not token:
        return False
    account = get_account(email)
    if account is None:
        return False
    if account.get("reset_token") != token:
        return False
    created_at = account.get("reset_token_created_at")
    return bool(created_at and datetime.now(timezone.utc) - created_at <= RESET_TOKEN_MAX_AGE)


def reset_password(email: str, token: str, new_password: str) -> None:
    """Sets a new password after verifying the reset token, and clears the
    token so it can't be reused (a reset link is meant to work once)."""
    if not verify_reset_token(email, token):
        raise AuthError("رابط إعادة التعيين غير صالح أو منتهي الصلاحية. يرجى طلب رابط جديد.")
    if len(new_password) < 8:
        raise AuthError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    email = email.strip().lower()
    password_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
    _accounts().document(email).set({
        "password_hash": password_hash,
        "reset_token": "",
    }, merge=True)


def update_profile(
    email: str, field_of_study: str, academic_level: str,
    tone: str = "", custom_instructions: str = "",
) -> None:
    """Saves the account's self-reported profile (field of study, academic
    level, preferred tone, free-text custom instructions). All optional --
    pass "" to clear any of them. academic_level must be "" or one of
    ACADEMIC_LEVELS; tone must be "" or one of TONE_OPTIONS;
    custom_instructions is capped at CUSTOM_INSTRUCTIONS_MAX_LEN."""
    if academic_level and academic_level not in ACADEMIC_LEVELS:
        raise AuthError(f"مستوى أكاديمي غير معروف: {academic_level}")
    if tone and tone not in TONE_OPTIONS:
        raise AuthError(f"أسلوب غير معروف: {tone}")
    custom_instructions = custom_instructions.strip()
    if len(custom_instructions) > CUSTOM_INSTRUCTIONS_MAX_LEN:
        raise AuthError(f"التعليمات طويلة جداً (الحد الأقصى {CUSTOM_INSTRUCTIONS_MAX_LEN} حرفاً).")
    email = email.strip().lower()
    _accounts().document(email).set({
        "field_of_study": field_of_study.strip(),
        "academic_level": academic_level,
        "tone": tone,
        "custom_instructions": custom_instructions,
    }, merge=True)


def _free_window_state(account: dict) -> tuple:
    """Returns (used, window_start) for the account's free-tier daily
    allowance, resetting to (0, now) if the 24h window has elapsed or never
    started. Read-only -- does not write to Firestore; increment_free_used()
    below writes the possibly-reset state back."""
    now = datetime.now(timezone.utc)
    window_start = account.get("free_window_start")
    used = account.get("free_used", 0)
    if window_start is None or now - window_start >= FREE_DAILY_WINDOW:
        return 0, now
    return used, window_start


def free_searches_remaining(account: dict) -> int:
    used, _ = _free_window_state(account)
    return max(0, FREE_DAILY_SEARCH_LIMIT - used)


def free_reset_hours_remaining(account: dict) -> float:
    """Hours until the free-tier allowance next resets. 0 if it already has
    (or never started -- nothing to wait for)."""
    _, window_start = _free_window_state(account)
    elapsed = datetime.now(timezone.utc) - window_start
    remaining = FREE_DAILY_WINDOW - elapsed
    return max(0.0, remaining.total_seconds() / 3600)


def increment_free_used(email: str) -> None:
    email = email.strip().lower()
    account = get_account(email) or {}
    used, window_start = _free_window_state(account)
    _accounts().document(email).set({
        "free_used": used + 1,
        "free_window_start": window_start,
    }, merge=True)


def increment_subscription_used(email: str) -> None:
    email = email.strip().lower()
    _accounts().document(email).set({"subscription_used": firestore.Increment(1)}, merge=True)


def is_subscribed(account: dict) -> bool:
    until = account.get("subscribed_until")
    if until is None:
        return False
    return until > datetime.now(timezone.utc)


def subscription_searches_remaining(account: dict) -> int:
    limit = account.get("subscription_search_limit", 0)
    used = account.get("subscription_used", 0)
    return max(0, limit - used)


def grant_subscription(email: str, days: int, plan: str = "normal") -> None:
    """
    Grants (or extends/renews) manually-paid subscription access for this
    account. Extends subscribed_until from the account's current value if
    that's still in the future (so renewing early doesn't lose remaining
    paid time), otherwise starts counting from now. Every grant/renewal
    resets the search allowance to a fresh per-plan limit (see
    SUBSCRIPTION_SEARCH_LIMITS) for the new period -- paying again means a
    new period's allowance, not indefinitely accumulating unused searches.

    plan selects which models the account's searches use for the harder
    reasoning stages (see run_assistant.PLAN_MODELS) -- "normal" is the same
    models every account already gets; "pro"/"max" cost more per search, so
    they also get a smaller search allowance (SUBSCRIPTION_SEARCH_LIMITS).
    """
    if plan not in PLANS:
        raise AuthError(f"خطة غير معروفة: {plan}")

    email = email.strip().lower()
    account = get_account(email)
    if account is None:
        raise AuthError("لا يوجد حساب بهذا البريد الإلكتروني.")

    now = datetime.now(timezone.utc)
    current_until = account.get("subscribed_until")
    start = current_until if (current_until and current_until > now) else now
    new_until = start + timedelta(days=days)
    _accounts().document(email).set({
        "subscribed_until": new_until,
        "subscription_search_limit": SUBSCRIPTION_SEARCH_LIMITS[plan],
        "subscription_used": 0,
        "plan": plan,
    }, merge=True)


def set_pending_wayl_payment(email: str, reference_id: str, plan: str) -> None:
    """
    Records a Wayl checkout that was just started, BEFORE the user is sent
    to Wayl's hosted checkout page. Needed because leaving this app for a
    different domain (Wayl's checkout) and coming back starts a fresh
    Streamlit session server-side -- st.session_state from before the trip
    does not survive it. Storing this on the account document (not
    session_state) is what lets the return trip know which reference id and
    plan to check, regardless of what Wayl's own redirect happens to append
    to the URL (found via real testing to be an opaque `orderid` value, not
    the referenceId this app generated -- so the URL itself can't be relied
    on for this).
    """
    email = email.strip().lower()
    _accounts().document(email).set({
        "pending_wayl_reference": reference_id,
        "pending_wayl_plan": plan,
    }, merge=True)


def get_pending_wayl_payment(email: str) -> dict | None:
    """Returns {"reference_id": ..., "plan": ...} if this account has a
    Wayl checkout awaiting confirmation, else None."""
    account = get_account(email)
    if not account:
        return None
    reference_id = account.get("pending_wayl_reference")
    plan = account.get("pending_wayl_plan")
    if not reference_id or plan not in PLANS:
        return None
    return {"reference_id": reference_id, "plan": plan}


def clear_pending_wayl_payment(email: str) -> None:
    email = email.strip().lower()
    _accounts().document(email).set({
        "pending_wayl_reference": firestore.DELETE_FIELD,
        "pending_wayl_plan": firestore.DELETE_FIELD,
    }, merge=True)
