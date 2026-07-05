import logging
from flask import current_app
import requests


def send_telegram_message(chat_id, text: str):
    """
    Sends a text message to the given Telegram chat using the configured bot token.

    Parameters
    ----------
    chat_id : int or str
        Telegram chat ID to send the message to (taken from the incoming update,
        so replies always go back to whoever messaged the bot).
    text : str
        Message body to send.
    """
    url = f"https://api.telegram.org/bot{current_app.config['TELEGRAM_BOT_TOKEN']}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to send Telegram message: {e}")
        return None

    logging.info(f"Status: {response.status_code}")
    return response


def process_telegram_message(body: dict, custom_response_function: callable = None):
    """
    Processes an incoming Telegram update and sends a response back to the same chat.

    Parameters
    ----------
    body : dict
        The incoming Telegram update payload (see https://core.telegram.org/bots/api#update).
    custom_response_function : callable, optional
        A custom function that maps the incoming message text to a response string.
        If not provided, falls back to a simple echo/uppercase response, matching
        the default behaviour of the WhatsApp integration.
    """
    message = body["message"]
    chat_id = message["chat"]["id"]

    try:
        message_text = message["text"]
    except KeyError:  # non-text message (e.g. sticker, photo)
        message_text = ""

    response = message_text.upper() if custom_response_function is None else custom_response_function(message_text)
    if response == "":  # empty response -> nothing to send
        return

    send_telegram_message(chat_id, response)


def is_valid_telegram_message(body: dict) -> bool:
    """
    Check if the incoming webhook payload is a Telegram update containing a text message.
    """
    return bool(
        body.get("message")
        and body["message"].get("chat")
        and "text" in body["message"]
    )
