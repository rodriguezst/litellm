"""
OAuth PKCE authenticator for Antigravity API.

This module implements the OAuth 2.0 PKCE flow for authenticating with
Google's Antigravity API. It uses a local callback server to receive
the authorization code.
"""
import base64
import hashlib
import http.server
import json
import os
import secrets
import socketserver
import threading
import time
import urllib.parse
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    ANTIGRAVITY_CALLBACK_PORT,
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    ANTIGRAVITY_DEFAULT_PROJECT,
    ANTIGRAVITY_REDIRECT_URI,
    ANTIGRAVITY_SCOPES,
    AuthenticationError,
    CallbackServerError,
    GOOGLE_AUTH_URL,
    GOOGLE_TOKEN_URL,
    GOOGLE_USERINFO_URL,
    ProjectDiscoveryError,
    TokenExpiredError,
    TokenRefreshError,
)


# Token expiry buffer (refresh 60 seconds before actual expiry)
TOKEN_EXPIRY_BUFFER_MS = 60 * 1000


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""

    def __init__(self, *args, callback_result: dict, callback_event: threading.Event, **kwargs):
        self.callback_result = callback_result
        self.callback_event = callback_event
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP logging."""
        pass

    def do_GET(self):
        """Handle GET request for OAuth callback."""
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/oauth-callback":
            query_params = urllib.parse.parse_qs(parsed_path.query)

            if "code" in query_params:
                self.callback_result["code"] = query_params["code"][0]
                self.callback_result["state"] = query_params.get("state", [None])[0]
                self._send_success_response()
            elif "error" in query_params:
                self.callback_result["error"] = query_params["error"][0]
                self.callback_result["error_description"] = query_params.get(
                    "error_description", ["Unknown error"]
                )[0]
                self._send_error_response()
            else:
                self.callback_result["error"] = "missing_code"
                self.callback_result["error_description"] = "No authorization code received"
                self._send_error_response()

            self.callback_event.set()
        else:
            self.send_error(404, "Not Found")

    def _send_success_response(self):
        """Send success HTML response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Authentication Successful</title></head>
        <body style="font-family: system-ui; text-align: center; padding-top: 50px;">
            <h1>✅ Authentication Successful!</h1>
            <p>You can close this window and return to your terminal.</p>
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def _send_error_response(self):
        """Send error HTML response."""
        self.send_response(400)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Authentication Failed</title></head>
        <body style="font-family: system-ui; text-align: center; padding-top: 50px;">
            <h1>❌ Authentication Failed</h1>
            <p>Please try again from your terminal.</p>
        </body>
        </html>
        """
        self.wfile.write(html.encode())


class Authenticator:
    """OAuth PKCE authenticator for Antigravity API."""

    def __init__(self) -> None:
        """Initialize the Antigravity authenticator with configurable token paths."""
        # Token storage paths
        self.token_dir = os.getenv(
            "ANTIGRAVITY_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/antigravity"),
        )
        self.token_file = os.path.join(
            self.token_dir,
            os.getenv("ANTIGRAVITY_TOKEN_FILE", "tokens.json"),
        )
        self._ensure_token_dir()

        # Cached tokens
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: int = 0
        self._project_id: Optional[str] = None
        self._email: Optional[str] = None

    def get_access_token(self) -> str:
        """
        Get a valid access token, refreshing or re-authenticating as needed.

        Returns:
            str: A valid access token.

        Raises:
            AuthenticationError: If unable to obtain an access token after retries.
        """
        # Try to load cached tokens
        if self._load_tokens():
            if not self._is_token_expired():
                return self._access_token  # type: ignore

            # Try to refresh
            try:
                verbose_logger.debug("Access token expired, attempting refresh")
                return self._refresh_access_token()
            except TokenRefreshError:
                verbose_logger.warning("Token refresh failed, re-authenticating")

        # Perform full OAuth flow
        for attempt in range(3):
            verbose_logger.debug(f"OAuth authentication attempt {attempt + 1}/3")
            try:
                return self._login()
            except (AuthenticationError, CallbackServerError) as e:
                verbose_logger.warning(f"Authentication attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    raise AuthenticationError(
                        message=f"Failed to authenticate after 3 attempts: {str(e)}",
                        status_code=401,
                    )
                continue

        raise AuthenticationError(
            message="Failed to get access token after 3 attempts",
            status_code=401,
        )

    def get_project_id(self) -> str:
        """
        Get the project ID for API requests.

        Returns:
            str: The project ID.
        """
        if self._project_id:
            return self._project_id

        self._load_tokens()
        return self._project_id or ANTIGRAVITY_DEFAULT_PROJECT

    def get_api_base(self) -> Optional[str]:
        """
        Get the API base URL from token storage.

        Returns:
            Optional[str]: The API base URL, or None if not set.
        """
        try:
            with open(self.token_file, "r") as f:
                data = json.load(f)
                return data.get("api_base")
        except (IOError, json.JSONDecodeError):
            return None

    def _is_token_expired(self) -> bool:
        """Check if the current access token is expired (with buffer)."""
        if not self._access_token or not self._expires_at:
            return True
        return self._expires_at <= (int(time.time() * 1000) + TOKEN_EXPIRY_BUFFER_MS)

    def _refresh_access_token(self) -> str:
        """
        Refresh the access token using the refresh token.

        Returns:
            str: The new access token.

        Raises:
            TokenRefreshError: If refresh fails.
        """
        if not self._refresh_token:
            raise TokenRefreshError(
                message="No refresh token available",
                status_code=401,
            )

        try:
            sync_client = _get_httpx_client()
            response = sync_client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": ANTIGRAVITY_CLIENT_ID,
                    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                error_code = error_data.get("error", "unknown")

                # Handle revoked token
                if error_code == "invalid_grant":
                    verbose_logger.warning("Refresh token revoked by Google, clearing tokens")
                    self._clear_tokens()
                    raise TokenRefreshError(
                        message="Refresh token has been revoked",
                        status_code=401,
                    )

                raise TokenRefreshError(
                    message=f"Token refresh failed: {error_code}",
                    status_code=response.status_code,
                )

            data = response.json()
            self._access_token = data["access_token"]
            self._expires_at = int(time.time() * 1000) + (data["expires_in"] * 1000)

            # Update refresh token if provided
            if "refresh_token" in data:
                self._refresh_token = data["refresh_token"]

            self._save_tokens()
            verbose_logger.info("Successfully refreshed access token")
            return self._access_token

        except httpx.HTTPError as e:
            raise TokenRefreshError(
                message=f"HTTP error during token refresh: {str(e)}",
                status_code=500,
            )

    def _login(self) -> str:
        """
        Perform full OAuth PKCE flow.

        Returns:
            str: The access token.

        Raises:
            AuthenticationError: If authentication fails.
        """
        # Generate PKCE verifier and challenge
        verifier, challenge = self._generate_pkce()

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)

        # Build authorization URL
        auth_params = {
            "client_id": ANTIGRAVITY_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
            "scope": " ".join(ANTIGRAVITY_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

        # Start callback server
        callback_result: Dict[str, Any] = {}
        callback_event = threading.Event()

        try:
            server = self._start_callback_server(callback_result, callback_event)
        except Exception as e:
            raise CallbackServerError(
                message=f"Failed to start callback server: {str(e)}",
                status_code=500,
            )

        # Display URL for user to authenticate
        print(  # noqa: T201
            f"\n🔐 Antigravity Authentication Required\n"
            f"Please visit the following URL to authenticate:\n\n"
            f"    {auth_url}\n\n"
            f"Waiting for authentication...",
            flush=True,
        )

        # Wait for callback (timeout after 5 minutes)
        callback_received = callback_event.wait(timeout=300)

        # Shutdown server
        server.shutdown()

        if not callback_received:
            raise AuthenticationError(
                message="Authentication timed out after 5 minutes",
                status_code=408,
            )

        if "error" in callback_result:
            raise AuthenticationError(
                message=f"OAuth error: {callback_result.get('error_description', callback_result['error'])}",
                status_code=400,
            )

        if "code" not in callback_result:
            raise AuthenticationError(
                message="No authorization code received",
                status_code=400,
            )

        # Verify state
        if callback_result.get("state") != state:
            raise AuthenticationError(
                message="State mismatch - possible CSRF attack",
                status_code=400,
            )

        # Exchange code for tokens
        return self._exchange_code(callback_result["code"], verifier)

    def _generate_pkce(self) -> Tuple[str, str]:
        """
        Generate PKCE code verifier and challenge.

        Returns:
            Tuple[str, str]: (verifier, challenge)
        """
        # Generate 32-byte random verifier
        verifier = secrets.token_urlsafe(32)

        # Create SHA256 challenge
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()

        return verifier, challenge

    def _start_callback_server(
        self, callback_result: dict, callback_event: threading.Event
    ) -> socketserver.TCPServer:
        """
        Start local HTTP server for OAuth callback.

        Returns:
            The server instance.
        """
        handler_class = lambda *args, **kwargs: OAuthCallbackHandler(
            *args, callback_result=callback_result, callback_event=callback_event, **kwargs
        )

        # Allow port reuse
        socketserver.TCPServer.allow_reuse_address = True
        server = socketserver.TCPServer(("localhost", ANTIGRAVITY_CALLBACK_PORT), handler_class)

        # Run server in background thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        return server

    def _exchange_code(self, code: str, verifier: str) -> str:
        """
        Exchange authorization code for tokens.

        Args:
            code: The authorization code from OAuth callback.
            verifier: The PKCE code verifier.

        Returns:
            str: The access token.

        Raises:
            AuthenticationError: If exchange fails.
        """
        try:
            sync_client = _get_httpx_client()
            response = sync_client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": ANTIGRAVITY_CLIENT_ID,
                    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
                    "code_verifier": verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                raise AuthenticationError(
                    message=f"Token exchange failed: {error_data.get('error_description', error_data.get('error', 'Unknown'))}",
                    status_code=response.status_code,
                )

            data = response.json()
            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token")
            self._expires_at = int(time.time() * 1000) + (data["expires_in"] * 1000)

            # Fetch user info
            self._fetch_user_info()

            # Discover project ID
            self._discover_project()

            # Save tokens
            self._save_tokens()

            print("✅ Authentication successful!", flush=True)  # noqa: T201
            verbose_logger.info(f"Authenticated as {self._email or 'unknown user'}")

            return self._access_token

        except httpx.HTTPError as e:
            raise AuthenticationError(
                message=f"HTTP error during token exchange: {str(e)}",
                status_code=500,
            )

    def _fetch_user_info(self) -> None:
        """Fetch user email from Google userinfo endpoint."""
        try:
            sync_client = _get_httpx_client()
            response = sync_client.get(
                f"{GOOGLE_USERINFO_URL}?alt=json",
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
            if response.status_code == 200:
                data = response.json()
                self._email = data.get("email")
        except Exception as e:
            verbose_logger.warning(f"Failed to fetch user info: {str(e)}")

    def _discover_project(self) -> None:
        """
        Discover project ID from Antigravity API.

        Falls back to default project if discovery fails.
        Uses production endpoint first per reference implementation.
        """
        from .common_utils import ANTIGRAVITY_LOAD_ENDPOINTS

        for endpoint in ANTIGRAVITY_LOAD_ENDPOINTS:
            try:
                sync_client = _get_httpx_client()
                response = sync_client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers={
                        "Authorization": f"Bearer {self._access_token}",
                        "Content-Type": "application/json",
                    },
                    json={},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    # Check for project ID in multiple possible fields
                    # Reference implementation looks for cloudaicompanionProject
                    project_id = None
                    if "cloudaicompanionProject" in data:
                        cap = data["cloudaicompanionProject"]
                        if isinstance(cap, str) and cap:
                            project_id = cap
                        elif isinstance(cap, dict) and cap.get("id"):
                            project_id = cap["id"]
                    elif "project" in data:
                        project_id = data["project"]

                    if project_id:
                        self._project_id = project_id
                        verbose_logger.info(f"Discovered project ID: {self._project_id}")
                        return

            except Exception as e:
                verbose_logger.debug(f"Project discovery failed on {endpoint}: {str(e)}")
                continue

        # Fallback to default
        self._project_id = ANTIGRAVITY_DEFAULT_PROJECT
        verbose_logger.warning(f"Using default project ID: {self._project_id}")

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, mode=0o700, exist_ok=True)

    def _save_tokens(self) -> None:
        """Save tokens to local storage."""
        try:
            data = {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "expires_at": self._expires_at,
                "project_id": self._project_id,
                "email": self._email,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self.token_file, "w") as f:
                json.dump(data, f, indent=2)
            # Set restrictive permissions
            os.chmod(self.token_file, 0o600)
        except IOError as e:
            verbose_logger.error(f"Failed to save tokens: {str(e)}")

    def _load_tokens(self) -> bool:
        """
        Load tokens from local storage.

        Returns:
            bool: True if tokens were successfully loaded.
        """
        try:
            with open(self.token_file, "r") as f:
                data = json.load(f)

            self._access_token = data.get("access_token")
            self._refresh_token = data.get("refresh_token")
            self._expires_at = data.get("expires_at", 0)
            self._project_id = data.get("project_id")
            self._email = data.get("email")

            return bool(self._access_token)

        except (IOError, json.JSONDecodeError) as e:
            verbose_logger.debug(f"No cached tokens found: {str(e)}")
            return False

    def _clear_tokens(self) -> None:
        """Clear cached tokens."""
        self._access_token = None
        self._refresh_token = None
        self._expires_at = 0
        self._project_id = None
        self._email = None

        try:
            if os.path.exists(self.token_file):
                os.remove(self.token_file)
        except IOError as e:
            verbose_logger.warning(f"Failed to remove token file: {str(e)}")
