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
        if not load_dotenv(chatbot_env_path): raise ValueError("Failed to load .env file")
        self._access_token = os.getenv("ACCESS_TOKEN")
        self._recipient_waid = os.getenv("RECIPIENT_WAID")
        self._phone_number_id = os.getenv("PHONE_NUMBER_ID")
        self._version = os.getenv("VERSION")
        self._app_id = os.getenv("APP_ID")
        self._app_secret = os.getenv("APP_SECRET")
        self._verify_token = os.getenv("VERIFY_TOKEN")  # from webhook's callback URL (ngrok) in meta developers
        self._media_id_list_file = os.getenv("MEDIA_ID_LIST_FILE")

        self.verbose = verbose

    def __call__(self, message: str):
        """
        Parameters
        ----------
        message : str
            The message to be sent using the `send_message` method.
        """
        self.send_message(message)

    def send_message(self, message: str, image_path: Union[str, Path] = None):
        """
        send_message method

        Sends a message with an optional image attachment.

        Parameters
        ----------
        message : str
            The text message to be sent.
        image_path : Union[str, Path], optional
            The file path to the image to be sent as an attachment. Defaults to None.

        """
        data = self._get_text_message_input(text=message, image_path=image_path)
        self._send_message_backend(data, verbose=self.verbose)

    # auxiliary methods:
    def _get_text_message_input(self, text: str, image_path: Union[str, Path] = None) -> str:
        """
        Creates a WhatsApp message payload for sending either a text message or an image with an optional caption.

        Parameters
        ----------
        text : str
            The text message to be sent. This is a required parameter and may also serve as a caption if an image is included.
        image_path : Union[str, Path], optional
            The file path to the image to be sent along with the message. When provided, the image is uploaded to generate a media ID for the message. Defaults to None.

        Returns
        -------
        str
            A JSON-formatted string containing the message payload for the WhatsApp API. Different payload structures are created for image and text messages.
        """
        # send image:
        if image_path is not None:
            # file details:
            image_path = image_path if isinstance(image_path, str) else str(image_path)

            # check whether file is already on server:
            if image_path in self.filepath_media_id_dict.keys():
                media_id = self.filepath_media_id_dict[image_path]

            else:  # upload necessary
                # prepare upload:
                dtype = image_path.split('.')[-1]
                file_name = image_path.split('/')[-1]
                url = f"https://graph.facebook.com/{self._version}/{self._phone_number_id}/media"
                headers = {
                    "Authorization": f"Bearer {self._access_token}",
                }
                files = {
                    'file': (file_name, open(image_path, 'rb'), f'image/{dtype}')
                }
                data = {
                    'messaging_product': 'whatsapp',  # meant for whatsapp sending
                    'type': f'image/{dtype}'
                }
                # request upload:
                response = requests.post(url, headers=headers, files=files, data=data)

                # request media id and prepare JSON:
                media_id = response.json()["id"]
                self.save_media_id(media_id, image_path)  # save media id

            return json.dumps(
                {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": self._recipient_waid,
                    "type": "image",
                    "image": {"id": media_id,
                              "caption": text},
                }
            )

        # only send text:
        return json.dumps(
            {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": self._recipient_waid,
                "type": "text",
                "text": {"preview_url": False, "body": text},
            }
        )

    def _send_message_backend(self, data: str, verbose=False):
        """
        Sends a message using the Facebook Graph API.

        Parameters
        ----------
        data : str
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
            if verbose:
                print("Status:", response.status_code)
                print("Content-type:", response.headers["content-type"])
                print("Body:", response.text)
            return response
        else:
            if verbose:
                print(response.status_code)
                print(response.text)
            return response

    def test_connection(self):
        url = f"https://graph.facebook.com/{self._version}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": "Bearer " + self._access_token,
            "Content-Type": "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "to": self._recipient_waid,
            "type": "template",
            "template": {"name": "hello_world", "language": {"code": "en_US"}},
        }
        response = requests.post(url, headers=headers, json=data)
        if self.verbose:
            print(response.status_code)
            print(response.text)

    def save_media_id(self, media_id: str, image_path: Union[str, Path]) -> None:
        if isinstance(image_path, Path): image_path = str(image_path)  # type conversion if required
        # save media id:
        with open(self._media_id_list_file, 'a') as f:
            f.write(media_id.strip() + ' --- ' + image_path.strip() + '\n')

    def remove_media_id(self, media_id: str):
        # Read all lines from the file
        with open(self._media_id_list_file, "r") as file:
            lines = file.readlines()

        # Filter out the line containing the media_id
        updated_lines = [line for line in lines if media_id not in line]

        # Check if the line was removed
        if len(updated_lines) == len(lines):
            print(f"Media ID {media_id} not found in the file.")
            return False

        # Write the updated lines back to the file
        with open(self._media_id_list_file, "w") as file:
            file.writelines(updated_lines)

        print(f"Line with Media ID {media_id} successfully removed.")
        return True

    def delete_uploaded_media(self, media_id: str = None, image_path: Union[str, Path] = None) -> bool:
        """
        Deletes an uploaded image or other media from the server.

        Args:
            media_id (str): The ID of the media to be deleted (received after upload).

        Returns:
            bool: True if the media was deleted successfully, False otherwise.
        """
        if media_id is None and image_path is None:
            raise AttributeError("Either media_id or image_path must be provided.")
        elif image_path is not None:
            try:
                media_id = self.filepath_media_id_dict[image_path]
            except KeyError:
                raise ValueError(f"Media ID for {image_path} not found.")

        # Prepare the URL and headers
        url = f"https://graph.facebook.com/{self._version}/{media_id}"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
        }

        # Send the DELETE request
        response = requests.delete(url, headers=headers)

        # Check the response status
        if response.status_code == 200:
            print("Media deleted successfully.")
            self.remove_media_id(media_id=media_id)
            return True
        else:
            print(f"Failed to delete media: {response.status_code}, {response.text}")
            return False

    @property
    def filepath_media_id_dict(self) -> {str: str}:
        temp_dict = {}
        with open(self._media_id_list_file, 'r') as f:
            for line in f.readlines():
                if line[0] == '#': continue
                media_id, file_path = line.split(" --- ")
                media_id = media_id.strip();
                file_path = file_path.strip()
                if file_path in temp_dict.keys():  # delete second media-id if duplicate file_path:
                    print(f"Duplicate file_path found: {file_path}. Deleting second media-id.")
                    self.delete_uploaded_media(media_id=media_id)
                else:
                    temp_dict[file_path] = media_id
        return temp_dict