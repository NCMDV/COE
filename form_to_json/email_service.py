import logging
import requests
from requests.exceptions import HTTPError

from irefer.repo import ServiceLinksRepo
from irefer.dto import EmailServiceDTO, TokenDto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, token_url, client_id, client_username, client_password, service_links_repo):
        self.TOKEN_URL = token_url
        self.CLIENT_ID = client_id
        self.CLIENT_USERNAME = client_username
        self.CLIENT_PASSWORD = client_password
        self.service_links_repo = service_links_repo

    def send_email(self, request):
        BASE_URI = self.service_links_repo.findByKey("email service").getLink()

        try:
            response = requests.post(BASE_URI, json=request)
            response.raise_for_status()
            result = response.json()

            logger.info("Email id: %s", result['id'])
            logger.info("Email status: %s", result['emailStatus'])
            return result['emailStatus']
        except HTTPError as http_err:
            logger.error(f'HTTP error occurred: {http_err}')
            return None

    def send_secured_email(self, request):
        token = self.get_token().token()
        BASE_URI = self.service_links_repo.findByKey("secured email service").getLink()

        try:
            response = requests.post(BASE_URI, headers={'Authorization': f'Bearer {token}'}, json=request)
            response.raise_for_status()
            result = response.json()

            logger.info("Email id: %s", result['id'])
            logger.info("Email status: %s", result['emailStatus'])
            return result['emailStatus']
        except HTTPError as http_err:
            logger.error(f'HTTP error occurred: {http_err}')
            return None

    def get_token(self):
        try:
            cred = {'email': self.CLIENT_USERNAME, 'password': self.CLIENT_PASSWORD}

            EMAIL_SERVICE_URL = self.service_links_repo.findByKey("email service token").getLink()

            response = requests.post(EMAIL_SERVICE_URL, json=cred)
            response.raise_for_status()
            token_data = response.json()

            return TokenDto(token_data['token'])
        except HTTPError as http_err:
            logger.error(f'HTTP error occurred: {http_err}')
            return None

# send here without token: https://email-sender-dev.ap-southeast-1.elasticbeanstalk.com/api/v1/email/send