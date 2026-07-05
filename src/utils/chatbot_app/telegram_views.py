import logging
import json

from flask import Blueprint, request, jsonify, current_app

from .decorators.security import telegram_secret_required
from .utils.telegram_utils import process_telegram_message, is_valid_telegram_message

telegram_blueprint = Blueprint("telegram_webhook", __name__)


def handle_telegram_message(custom_response_function: callable = None):
    """
    Handles incoming Telegram webhook updates, validates them, and dispatches to the
    same custom response function used by the WhatsApp webhook (set via
    `app.config["CUSTOM_RESPONSE_FUNCTION"]`), so both channels talk to the same
    request-map dispatch in the workflow process.

    Parameters
    ----------
    custom_response_function : callable, optional
        A custom function provided to handle Telegram message responses. If not
        provided, the default echo/uppercase logic in `telegram_utils` is used.

    Returns
    -------
    flask.Response
        A JSON response indicating the status of the request, with an appropriate
        HTTP status code:
        - 200: message processed successfully.
        - 404: payload is not a recognised Telegram text-message update.
        - 400: JSON body could not be decoded.
    """
    body = request.get_json()

    try:
        if is_valid_telegram_message(body):
            process_telegram_message(body, custom_response_function=custom_response_function)
            return jsonify({"status": "ok"}), 200
        else:
            # non-text updates (stickers, edited messages, etc.) are simply ignored
            return jsonify({"status": "error", "message": "Not a valid Telegram text message"}), 404
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON")
        return jsonify({"status": "error", "message": "Invalid JSON provided"}), 400


@telegram_blueprint.route("/telegram-webhook", methods=["POST"])
@telegram_secret_required
def telegram_webhook_post():
    try:
        custom_func = current_app.config['CUSTOM_RESPONSE_FUNCTION']
    except KeyError:
        custom_func = None
    return handle_telegram_message(custom_response_function=custom_func)
