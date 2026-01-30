import json
import requests
import os
import smtplib
import mimetypes
import re
from pathlib import Path
from typing import Union, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

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
        if self.verbose:
            print("Sending message: {}".format(message))
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


class MailChatbot:
    """
    Class MailChatbot

    Manages sending emails via SMTP using standard mail protocols. It loads necessary
    configuration from an environment file and provides methods to send text messages
    with optional attachments.

    Parameters
    ----------
    chatbot_env_path : Union[str, Path]
        Path to the environment file (.env) containing necessary mail server credentials
        and configuration.
    verbose : bool, optional
        If True, enables verbose logging for debugging. Default is False.
    max_attachment_size_mb : int, optional
        Maximum attachment size in MB. Default is 25 (typical SMTP limit).

    Raises
    ------
    ValueError
        If loading the environment file fails or required credentials are missing.

    Attributes
    ----------
    _smtp_host : str
        SMTP server hostname (e.g., 'smtp.gmail.com', 'mail.example.com').
    _smtp_port : int
        SMTP server port (typically 587 for TLS, 465 for SSL).
    _smtp_timeout : int
        Timeout in seconds for SMTP operations.
    _sender_email : str
        Email address of the sender (usually the authenticated account).
    _sender_password : str
        Password or app-specific password for SMTP authentication.
    _recipient_email : Union[str, List[str]]
        Email address(es) of the recipient(s).
    _smtp_use_tls : bool
        Whether to use TLS encryption for the connection. Default is True.
    verbose : bool
        Indicates whether verbose logging is enabled.
    max_attachment_size_mb : int
        Maximum allowed attachment size in MB.
    """

    # Regex pattern for basic email validation
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    def __init__(
            self,
            chatbot_env_path: Union[str, Path],
            verbose: bool = False,
            max_attachment_size_mb: int = 25,
            smtp_timeout: int = 10
    ):
        """
        Initialize MailChatbot by loading environment variables from .env file.

        Parameters
        ----------
        chatbot_env_path : Union[str, Path]
            Path to the .env configuration file.
        verbose : bool, optional
            Enable verbose logging. Default is False.
        max_attachment_size_mb : int, optional
            Maximum attachment size allowed in MB. Default is 25.
        smtp_timeout : int, optional
            SMTP operation timeout in seconds. Default is 10.

        Raises
        ------
        ValueError
            If .env file cannot be loaded or required variables are missing.
        """
        if not load_dotenv(chatbot_env_path):
            raise ValueError("Failed to load .env file")

        # Load SMTP configuration
        self._smtp_host = os.getenv("SMTP_HOST")


        try:
            self._smtp_port = int(os.getenv("SMTP_PORT", "587"))
        except ValueError:
            raise ValueError("SMTP_PORT environment variable must be a valid integer")

        self._smtp_timeout = smtp_timeout
        self._sender_email = os.getenv("SENDER_EMAIL")
        self._sender_password = os.getenv("SENDER_PASSWORD")
        self._recipient_email = os.getenv("RECIPIENT_EMAIL")
        if verbose:
            print(f"Initializing MailChatbot sending from {self._sender_email} to {self._recipient_email}")
        self._smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

        # Validate required fields
        required_fields = [self._smtp_host, self._sender_email,
                           self._sender_password, self._recipient_email]
        if not all(required_fields):
            raise ValueError(
                "Missing required environment variables: "
                "SMTP_HOST, SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL"
            )

        # FIX: Validate email addresses
        if not self._validate_email(self._sender_email):
            raise ValueError(f"Invalid sender email format: {self._sender_email}")
        if not self._validate_email(self._recipient_email):
            raise ValueError(f"Invalid recipient email format: {self._recipient_email}")

        self.verbose = verbose
        self.max_attachment_size_mb = max_attachment_size_mb

    @staticmethod
    def _validate_email(email: str) -> bool:
        """
        Validate email address format.

        Parameters
        ----------
        email : str
            Email address to validate.

        Returns
        -------
        bool
            True if valid email format, False otherwise.
        """
        return MailChatbot.EMAIL_PATTERN.match(email) is not None

    def __call__(
            self,
            message: str,
            attachment_path: Union[str, Path] = None,
            subject: str = "Message"
    ) -> bool:
        """
        Make instance callable.

        Parameters
        ----------
        message : str
            The message body to be sent.
        attachment_path : Union[str, Path], optional
            Path to a file to attach to the email.
        subject : str, optional
            Email subject line. Default is "Message".

        Returns
        -------
        bool
            True if email sent successfully, False otherwise.
        """
        return self.send_message(message, attachment_path=attachment_path, subject=subject)

    def send_message(
            self,
            message: str,
            attachment_path: Union[str, Path] = None,
            subject: str = "TradingBot: Info",
            recipients: Optional[Union[str, List[str]]] = None
    ) -> bool:
        """
        send_message method

        Sends an email message with optional attachments.

        Parameters
        ----------
        message : str
            The text body of the email.
        attachment_path : Union[str, Path], optional
            Path to a single file to attach. Default is None.
        subject : str, optional
            Email subject line. Default is "Message".
        recipients : Optional[Union[str, List[str]]], optional
            Email recipient(s). If None, uses RECIPIENT_EMAIL from .env.

        Returns
        -------
        bool
            True if email sent successfully, False otherwise.
        """
        if self.verbose:
            print(f"Preparing email message: {subject}")

        try:
            # Build email payload
            email_payload = self._get_email_message_input(
                text=message,
                attachment_path=attachment_path,
                subject=subject,
                recipients=recipients
            )

            # FIX: Return and propagate the result
            return self._send_message_backend(email_payload, verbose=self.verbose)

        except FileNotFoundError as e:
            if self.verbose:
                print(f"Error: Attachment file not found: {str(e)}")
            return False
        except ValueError as e:
            if self.verbose:
                print(f"Error: {str(e)}")
            return False

    def _get_email_message_input(
            self,
            text: str,
            attachment_path: Union[str, Path] = None,
            subject: str = "Message",
            recipients: Optional[Union[str, List[str]]] = None
    ) -> MIMEMultipart:
        """
        Creates an email message payload with optional attachments.

        Parameters
        ----------
        text : str
            The main text body of the email.
        attachment_path : Union[str, Path], optional
            Path to a file to attach. Default is None.
        subject : str, optional
            Email subject line. Default is "Message".
        recipients : Optional[Union[str, List[str]]], optional
            Recipient email address(es). If None, uses configured recipient.

        Returns
        -------
        MIMEMultipart
            Email message object ready to send via SMTP.

        Raises
        ------
        FileNotFoundError
            If attachment path does not exist.
        ValueError
            If recipient email format is invalid.
        """
        # Determine recipients
        if recipients is None:
            recipients = self._recipient_email
        if isinstance(recipients, str):
            recipients = [recipients]

        # Validate recipients
        for recipient in recipients:
            if not self._validate_email(recipient):
                raise ValueError(f"Invalid recipient email format: {recipient}")

        # Create multipart message container
        msg = MIMEMultipart()
        msg["From"] = self._sender_email
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        # Attach text body
        msg.attach(MIMEText(text, "plain"))

        # Attach file if provided
        if attachment_path is not None:
            self._attach_file(msg, attachment_path)

        return msg

    def _attach_file(self, msg: MIMEMultipart, file_path: Union[str, Path]) -> None:
        """
        Attaches a file to an email message.

        Parameters
        ----------
        msg : MIMEMultipart
            The email message object.
        file_path : Union[str, Path]
            Path to the file to attach.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If file size exceeds maximum allowed attachment size.
        """
        file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

        if not file_path.exists():
            raise FileNotFoundError(f"Attachment file not found: {file_path}")

        # FIX: Validate file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_attachment_size_mb:
            raise ValueError(
                f"Attachment size ({file_size_mb:.2f}MB) exceeds maximum "
                f"({self.max_attachment_size_mb}MB)"
            )

        file_name = file_path.name

        # FIX: Use mimetypes for proper MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        main_type, sub_type = mime_type.split("/")

        # FIX: Use appropriate MIME type for text files
        if main_type == "text":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as attachment:
                part = MIMEText(attachment.read(), _subtype=sub_type)
        else:
            # Binary files: read as bytes and encode
            with open(file_path, "rb") as attachment:
                part = MIMEBase(main_type, sub_type)
                part.set_payload(attachment.read())
                encoders.encode_base64(part)

        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {file_name}",
        )
        msg.attach(part)

        if self.verbose:
            print(f"Attached file: {file_name} ({file_size_mb:.2f}MB)")

    def _send_message_backend(self, msg: MIMEMultipart, verbose: bool = False) -> bool:
        """
        Sends an email using SMTP.

        Parameters
        ----------
        msg : MIMEMultipart
            The email message object to send.
        verbose : bool, optional
            If True, prints debug information. Default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # FIX: Use try-finally to ensure connection closure
        server = None
        try:
            # Establish SMTP connection with timeout
            if self._smtp_use_tls:
                server = smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._smtp_timeout)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self._smtp_host, self._smtp_port, timeout=self._smtp_timeout)

            # Authenticate
            server.login(self._sender_email, self._sender_password)

            # FIX: Remove unused recipients extraction
            # smtplib.send_message() automatically extracts recipients from headers
            server.send_message(msg)

            if verbose:
                print(f"Status: Email sent successfully")
                print(f"To: {msg['To']}")
                print(f"Subject: {msg['Subject']}")

            return True

        except smtplib.SMTPAuthenticationError:
            if verbose:
                print("Error: SMTP Authentication failed. Check sender email and password.")
            return False

        except smtplib.SMTPNotSupportedError:
            if verbose:
                print("Error: SMTP server does not support TLS/SSL.")
            return False

        except smtplib.SMTPException as e:
            if verbose:
                print(f"SMTP Error: {str(e)}")
            return False

        except TimeoutError:
            if verbose:
                print(f"Error: SMTP connection timeout after {self._smtp_timeout}s")
            return False

        except OSError as e:
            if verbose:
                print(f"Error: Network error - {str(e)}")
            return False

        finally:
            # FIX: Ensure connection is properly closed
            if server:
                try:
                    server.quit()
                except Exception:
                    # Suppress errors during quit (connection may be already closed)
                    pass

    def test_connection(self) -> bool:
        """
        Tests the SMTP connection and authentication.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        print("Testing SMTP connection...")
        server = None
        try:
            # FIX: Use timeout here too
            if self._smtp_use_tls:
                server = smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._smtp_timeout)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self._smtp_host, self._smtp_port, timeout=self._smtp_timeout)

            server.login(self._sender_email, self._sender_password)

            if self.verbose:
                print("✓ SMTP connection test successful")
            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ SMTP connection test failed: {str(e)}")
            return False

        finally:
            # FIX: Ensure connection is closed
            if server:
                try:
                    server.quit()
                except Exception:
                    pass


# Example usage in __main__
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = MailChatbot("../../private/chatbot.env", verbose=True, smtp_timeout=30)

    # Test connection
    success = chatbot.test_connection()
    if not success:
        print("Connection test failed. Check your credentials.")
        exit(1)

    # Send simple message
    result = chatbot.send_message("Hello! This is a test email.")
    if result:
        print("Message sent successfully!")
    else:
        print("Failed to send message")

    # Send message with subject
    result = chatbot(
        "See the new prediction.",
        subject="Test Prediction Plot Mail",
        attachment_path="../../output/prediction_plots/2025-09-18 13_11_12 Prediction Visualisation.png"
    )
    if result:
        print("Report sent!")

"""
if __name__ == "__main__":
    chatter = WhatsAppChatbot("../../private/chatbot.env", verbose=True)
    chatter.test_connection()"""