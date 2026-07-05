from functools import wraps
from flask import current_app, jsonify, request
import logging
import hashlib
import hmac


def validate_signature(payload, signature):
    """
    Validate the incoming payload's signature against our expected signature
    """
    # Use the App Secret to hash the payload
    expected_signature = hmac.new(
        bytes(current_app.config["APP_SECRET"], "latin-1"),
        msg=payload.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    # Check if the signature matches
    return hmac.compare_digest(expected_signature, signature)


def signature_required(f):
    """
    Decorator to ensure that the incoming requests to our webhook are valid and signed with the correct signature.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        signature = request.headers.get("X-Hub-Signature-256", "")[
            7:
        ]  # Removing 'sha256='
        if not validate_signature(request.data.decode("utf-8"), signature):
            logging.info("Signature verification failed!")
            return jsonify({"status": "error", "message": "Invalid signature"}), 403
        return f(*args, **kwargs)

    return decorated_function


def telegram_secret_required(f):
    """
    Decorator to ensure that incoming requests to our Telegram webhook carry the
    secret token configured via `TelegramChatbot.set_webhook`, so we can verify the
    request actually originated from Telegram's servers (Telegram's equivalent of
    Meta's HMAC signature check).
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        configured_secret = current_app.config.get("TELEGRAM_WEBHOOK_SECRET")
        if configured_secret:  # only enforce if a secret has actually been configured
            received_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if not hmac.compare_digest(received_secret, configured_secret):
                logging.info("Telegram secret token verification failed!")
                return jsonify({"status": "error", "message": "Invalid secret token"}), 403
        return f(*args, **kwargs)

    return decorated_function
