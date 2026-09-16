"""
Talks to the Wayl payment API -- Python's built-in urllib, no new dependency,
same stdlib-only approach as email_sender.py. This module fails LOUD
(WaylClientError is always raised, never swallowed) -- unlike the fail-open
moderation/verification calls elsewhere in this project, a payment check must
never be silently treated as successful just because a request errored.

Required Streamlit secrets:
    WAYL_API_KEY -- the merchant's X-WAYL-AUTHENTICATION token (same value
                    works for both env="test" and env="live" requests --
                    the sandbox/live switch is a body field, not a separate
                    key, per Wayl's own API docs).

Real API reference (Wayl dashboard -> API Platform, read directly, not
guessed): base URL https://api.thewayl.com/api/v1, auth header
"X-WAYL-AUTHENTICATION: <token>".
"""

import json
import urllib.error
import urllib.request

API_BASE = "https://api.thewayl.com/api/v1"


class WaylClientError(Exception):
    """Raised on any failure talking to Wayl -- callers must treat this as
    "payment status unknown," never as "payment failed" or "payment
    succeeded." Only an explicit status value from a successful response
    means anything."""


def _request(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-WAYL-AUTHENTICATION": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise WaylClientError(f"Wayl API returned {error.code} for {method} {path}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise WaylClientError(f"Wayl API request failed for {method} {path}: {error}") from error


def create_payment_link(
    api_key: str,
    reference_id: str,
    amount_iqd: int,
    label: str,
    redirection_url: str,
    env: str = "live",
) -> str:
    """
    Creates a Wayl hosted-checkout link and returns its URL (the response's
    data.url field, per Wayl's docs). reference_id must be unique per
    checkout attempt -- the caller is responsible for that (e.g. include the
    account email + plan + a timestamp/uuid).

    env="test" uses Wayl's sandbox mode (same API key, per their docs --
    confirmed on the 2026-09-12 demo call that a test mode exists). Always
    use env="test" until a real end-to-end sandbox payment has been
    verified; switch to "live" only after that.

    webhookUrl/webhookSecret are deliberately omitted -- this app has no way
    to receive an incoming webhook (Streamlit can't expose a custom HTTP
    endpoint), so it relies on redirection_url + polling (check_payment_status)
    instead. See the Wayl integration plan for why.
    """
    body = {
        "env": env,
        "referenceId": reference_id,
        "total": amount_iqd,
        "currency": "IQD",
        "customParameter": "",
        "lineItem": [
            {"label": label, "amount": amount_iqd, "type": "increase"},
        ],
        "redirectionUrl": redirection_url,
    }
    result = _request("POST", "/links", api_key, body)
    try:
        return result["data"]["url"]
    except (KeyError, TypeError) as error:
        raise WaylClientError(f"Unexpected response shape from Wayl create-link: {result}") from error


def get_payment_status(api_key: str, reference_id: str) -> dict:
    """
    Returns the raw status dict for a payment link (status, paymentMethod,
    completedAt, per Wayl's docs). Callers decide what counts as "paid" --
    this function does not interpret the status string itself, since the
    exact set of values Wayl actually returns hasn't been observed yet
    (confirm against a real sandbox response before relying on a specific
    string match).
    """
    return _request("GET", f"/links/{reference_id}", api_key, body=None)
