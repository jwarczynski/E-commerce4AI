import base64
import datetime
import hashlib
import time
from typing import Any

import jwt
import requests
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from cafe.utils.config import load_config
from cafe.utils.logger import setup_logger


class SnowflakeClient:
    """Singleton class for managing Snowflake connection and JWT auth."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SnowflakeClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the Snowflake connection using key pair authentication."""
        self.logger = setup_logger(__name__)
        self.config = load_config()["snowflake"]

        # Load private key
        with open(self.config["private_key_path"], "rb") as key_file:
            pswd = self.config.get("private_key_passphrase").encode() if self.config.get(
                "private_key_passphrase"
            ) else None
            p_key = serialization.load_pem_private_key(
                key_file.read(),
                password=pswd,
                backend=default_backend()
            )

        self.private_key = p_key
        self.conn = snowflake.connector.connect(
            user=self.config["user"],
            account=self.config["account"],
            private_key=self.private_key,
            warehouse=self.config["warehouse"],
            role=self.config["role"]
        )
        self.logger.info("Snowflake connection initialized with key pair.")

        # Store the token and its expiration time
        self._jwt_token = None
        self._jwt_token_expiry = None

    def _generate_jwt_token(self) -> str:
        """Generate JWT token for Snowflake REST API (Cortex tool)."""
        org = self.config["organization"]
        account = self.config["account"]
        user = self.config["user"]

        # Account identifiers: replace dots with dashes if needed
        account_identifier = f"{org.upper()}-{account.upper()}".replace(".", "-")
        qualified_username = f"{account_identifier}.{user.upper()}"

        # Compute the SHA256 fingerprint of the public key (Snowflake format: base64)
        public_key_der = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        sha256_digest = hashlib.sha256(public_key_der).digest()
        public_key_fingerprint = base64.b64encode(sha256_digest).decode("utf-8")

        issuer = f"{qualified_username}.SHA256:{public_key_fingerprint}"

        now = datetime.datetime.now(datetime.timezone.utc)
        lifetime = datetime.timedelta(minutes=59)

        payload = {
            "iss": issuer,
            "sub": qualified_username,
            "iat": now,
            "exp": now + lifetime,
        }

        token = jwt.encode(payload, self.private_key, algorithm="RS256")

        # Ensure it's a str (PyJWT < 2.0 returns bytes)
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        decoded_token = jwt.decode(token, key=self.private_key.public_key(), algorithms=["RS256"])
        self.logger.info("Generated a JWT with the following payload:\n{}".format(decoded_token))

        return token

    def get_jwt_token(self) -> str:
        """Get the JWT token, generating a new one if expired."""
        if self._jwt_token and time.time() < self._jwt_token_expiry:
            return self._jwt_token

        self._jwt_token = self._generate_jwt_token()
        self._jwt_token_expiry = int(time.time()) + 3600  # Set expiry to 1 hour from now
        return self._jwt_token

    def execute_query(self, query: str) -> dict[str, Any]:
        """Execute a SQL query and return results."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("USE WAREHOUSE {}".format(self.config["warehouse"]))
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return {"columns": columns, "data": results}
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            cursor.close()

    def call_cortex_llm(self, data: dict[str, Any]) -> dict:
        """Call the Cortex tool API with a prompt and return the response."""
        snowflake_host = self.config["host"]
        url = f"https://{snowflake_host}/api/v2/cortex/inference:complete"
        jwt_token = self.get_jwt_token()

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def call_cortex_analyst(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Send a message to Cortex Analyst API."""

        host = self.config["host"]
        resp = requests.post(
            url=f"https://{host}/api/v2/cortex/analyst/message",
            json=request_body,
            headers={
                "Authorization": f"Bearer {self.get_jwt_token()}",
                "Content-Type": "application/json",
            },
        )
        request_id = resp.headers.get("X-Snowflake-Request-Id")
        if resp.status_code < 400:
            self.logger.debug(f"Response from Cortex Analyst: {resp.json()}")
            return {**resp.json(), "request_id": request_id}
        else:
            raise Exception(f"Failed request (id: {request_id}): {resp.text}")
