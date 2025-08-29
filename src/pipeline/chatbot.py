import json
import os
import requests
from dotenv import load_dotenv

from typing import Union
from pathlib import Path

import logging
from src.utils.chatbot_app import create_app

class WhatsAppChatbot:
    """
    Class WhatsAppChatbot
    Manages sending messages via WhatsApp using the Facebook Graph API. It loads necessary configuration from an environment file and provides methods to send text messages.

    Parameters
    ----------
    chatbot_env_path : Union[str, Path]
        Path to the environment file (.env) containing necessary API credentials and configuration.
    verbose : bool, optional
        If True, enables verbose logging for debugging. Default is False.

    Raises
    ------
    ValueError
        If loading the environment file fails.

    Attributes
    ----------
    _access_token : str
        Access token required for authenticating API requests.
    _recipient_waid : str
        WhatsApp ID of the message recipient.
    _phone_number_id : str
        Phone number ID linked with the WhatsApp Business API.
    _version : str
        API version to be used for the requests.
    _app_id : str
        Application ID for the API integration.
    _app_secret : str
        Application secret for authentication.
    _verify_token : str
        Token used to verify webhook integration with the API.
    verbose : bool
        Indicates whether verbose logging is enabled.
    """
    def __init__(self, chatbot_env_path: Union[str, Path], verbose: bool = False):
        # load env and respective vars:
        if not load_dotenv(chatbot_env_path): raise ValueError("Failed to load .env file")
        self._access_token = os.getenv("ACCESS_TOKEN")
        self._recipient_waid = os.getenv("RECIPIENT_WAID")
        self._phone_number_id = os.getenv("PHONE_NUMBER_ID")
        self._version = os.getenv("VERSION")
        self._app_id = os.getenv("APP_ID")
        self._app_secret = os.getenv("APP_SECRET")
        self._verify_token = os.getenv("VERIFY_TOKEN")  # from webhook's callback URL (ngrok) in meta developers

        self.verbose = verbose

    def __call__(self, message: str):
        """
        Parameters
        ----------
        message : str
            The message to be sent using the `send_message` method.
        """
        self.send_message(message)

    def send_message(self, message: str):
        """
        Sends a text message to a specified recipient.

        Parameters
        ----------
        message : str
            The text content of the message to be sent.
        """
        data = self._get_text_message_input(recipient=self._recipient_waid, text=message)
        self._send_message_backend(data)

    # auxiliary methods:
    @staticmethod
    def _get_text_message_input(recipient, text):
        """
        Prepare data for _send_message method.

        Parameters
        ----------
        recipient : str
            The phone number or unique identifier of the message recipient.
        text : str
            The body of the text message to be sent.
        """
        return json.dumps(
            {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": "text",
                "text": {"preview_url": False, "body": text},
            }
        )

    def _send_message_backend(self, data):
        """
        Sends a message using the Facebook Graph API.

        Parameters
        ----------
        data : dict
            The JSON payload to be sent in the HTTP POST request body.
        verbose : bool, optional
            If True, prints the status code, response headers, and response body for debugging. Default is False.

        Returns
        -------
        response : requests.Response
            The HTTP response object returned by the API call.
        """
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }

        url = f"https://graph.facebook.com/{self._version}/{self._phone_number_id}/messages"

        response = requests.post(url, data=data, headers=headers)
        if response.status_code == 200:
            if self.verbose:
                print("Status:", response.status_code)
                print("Content-type:", response.headers["content-type"])
                print("Body:", response.text)
            return response
        else:
            if self.verbose:
                print(response.status_code)
                print(response.text)
            return response