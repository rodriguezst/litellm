"""
Antigravity OAuth Authenticator for LiteLLM.

Implements the EXACT OAuth 2.0 Authorization Code flow with PKCE used by the
opencode-antigravity-auth reference implementation.

Flow:
1. Generate PKCE code verifier and challenge
2. Open browser to Google OAuth with all required scopes
3. Local server on localhost:51121 captures callback
4. Exchange authorization code for tokens
5. Discover project ID via loadCodeAssist API
6. Store refresh token with project context

Reference: https://github.com/NoeFabris/opencode-antigravity-auth
"""

import base64
import hashlib
import json
import os
import secrets
import socket
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    AntigravityAuthError,
    GetAccessTokenError,
    GetRefreshTokenError,
    TokenExpiredError,
)


# =============================================================================
# OAuth Configuration - EXACTLY as in reference implementation
# =============================================================================

OAUTH_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
OAUTH_CLIENT_SECRET = "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"

# OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Callback configuration
REDIRECT_PORT = 51121
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/oauth-callback"

# Required OAuth scopes (exactly as in reference)
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# Antigravity API endpoints for project discovery
ANTIGRAVITY_ENDPOINTS = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
]

# Default project ID (fallback)
DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# Callback timeout (5 minutes as in reference)
CALLBACK_TIMEOUT = 300


# =============================================================================
# PKCE Implementation
# =============================================================================

def generate_pkce_pair() -> Tuple[str, str]:
    """
    Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (verifier, challenge)
    """
    # Generate random verifier (43-128 characters)
    verifier = secrets.token_urlsafe(64)[:128]

    # Create S256 challenge
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")

    return verifier, challenge


def encode_state(verifier: str, project_id: str = "") -> str:
    """Encode OAuth state as base64url JSON."""
    payload = {"verifier": verifier, "projectId": project_id}
    return base64.urlsafe_b64encode(
        json.dumps(payload).encode("utf-8")
    ).decode("ascii").rstrip("=")


def decode_state(state: str) -> Dict[str, str]:
    """Decode OAuth state from base64url JSON."""
    # Add padding if needed
    padding = 4 - len(state) % 4
    if padding != 4:
        state += "=" * padding

    try:
        payload = json.loads(base64.urlsafe_b64decode(state))
        return payload
    except Exception:
        return {"verifier": "", "projectId": ""}


# =============================================================================
# OAuth Callback Handler
# =============================================================================

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback on localhost:51121."""

    # Class-level storage for callback data
    authorization_code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        pass

    def do_GET(self):
        """Handle GET request from OAuth callback."""
        parsed = urlparse(self.path)

        if parsed.path != "/oauth-callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.authorization_code = params["code"][0]
            OAuthCallbackHandler.state = params.get("state", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            # Success page (matches reference implementation style)
            html = """<!DOCTYPE html>
<html>
<head>
    <title>Antigravity Authentication</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            text-align: center;
            padding: 40px;
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }
        h1 { font-size: 2em; margin-bottom: 10px; }
        p { font-size: 1.2em; opacity: 0.9; }
        .checkmark { font-size: 4em; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="checkmark">✓</div>
        <h1>All set!</h1>
        <p>Authentication successful. You can close this tab.</p>
    </div>
    <script>setTimeout(function(){ window.close(); }, 3000);</script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif "error" in params:
            OAuthCallbackHandler.error = params.get("error_description", params["error"])[0]

            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            error_msg = OAuthCallbackHandler.error
            html = f"""<!DOCTYPE html>
<html>
<head><title>Authentication Failed</title></head>
<body style="font-family: sans-serif; text-align: center; padding: 50px;">
    <h1>❌ Authentication Failed</h1>
    <p>{error_msg}</p>
    <p>Please try again.</p>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(400)
            self.end_headers()


# =============================================================================
# Main Authenticator Class
# =============================================================================

class AntigravityAuthenticator:
    """
    OAuth authenticator for Google's Antigravity API.

    Implements the exact flow from opencode-antigravity-auth:
    1. PKCE-secured OAuth 2.0 Authorization Code flow
    2. Local callback server on localhost:51121
    3. Project discovery via loadCodeAssist API
    4. Multi-account support with token storage
    """

    def __init__(self) -> None:
        """Initialize authenticator with token storage paths."""
        self.token_dir = os.getenv(
            "ANTIGRAVITY_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/antigravity"),
        )
        self.accounts_file = os.path.join(self.token_dir, "accounts.json")
        self._ensure_token_dir()

    def get_api_key(self) -> str:
        """
        Get a valid access token, authenticating if necessary.

        This is the main entry point. It will:
        1. Check for existing valid access token
        2. Refresh if expired using refresh token
        3. Initiate OAuth flow if no valid tokens

        Returns:
            str: Valid OAuth access token for Antigravity API
        """
        # Try to get existing account
        account = self._get_active_account()

        if account:
            access_token = account.get("access_token")
            expires_at = account.get("expires_at", 0)

            # Check if token is still valid (with 60s buffer)
            if access_token and expires_at > time.time() + 60:
                return access_token

            # Try to refresh
            refresh_token = account.get("refresh_token")
            if refresh_token:
                try:
                    return self._refresh_access_token(account)
                except Exception as e:
                    verbose_logger.warning(f"Token refresh failed: {e}")

        # No valid token - need to authenticate
        return self._authenticate()

    def _authenticate(self) -> str:
        """
        Run the OAuth authentication flow.

        Opens browser for Google sign-in, captures callback, exchanges tokens.
        """
        # Generate PKCE pair
        verifier, challenge = generate_pkce_pair()

        # Encode state
        state = encode_state(verifier, "")

        # Build authorization URL
        auth_params = {
            "client_id": OAUTH_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": " ".join(OAUTH_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(auth_params)}"

        # Reset handler state
        OAuthCallbackHandler.authorization_code = None
        OAuthCallbackHandler.state = None
        OAuthCallbackHandler.error = None

        # Check if port is available
        if not self._is_port_available(REDIRECT_PORT):
            raise GetAccessTokenError(
                message=f"Port {REDIRECT_PORT} is already in use. Please close other applications using this port.",
                status_code=500,
            )

        # Start local server
        server = HTTPServer(("127.0.0.1", REDIRECT_PORT), OAuthCallbackHandler)
        server.timeout = 5  # Short timeout for polling

        # Print authentication message
        print(  # noqa: T201
            f"\n{'='*60}\n"
            f"ANTIGRAVITY AUTHENTICATION\n"
            f"{'='*60}\n"
            f"\nOpening browser for Google sign-in...\n"
            f"\nIf the browser doesn't open, please visit:\n"
            f"{auth_url}\n"
            f"\n{'='*60}\n",
            flush=True,
        )

        # Try to open browser
        try:
            webbrowser.open(auth_url)
        except Exception as e:
            verbose_logger.warning(f"Could not open browser: {e}")

        # Wait for callback (5 minute timeout)
        start_time = time.time()
        while (OAuthCallbackHandler.authorization_code is None
               and OAuthCallbackHandler.error is None):

            if time.time() - start_time > CALLBACK_TIMEOUT:
                server.server_close()
                raise GetAccessTokenError(
                    message="Authentication timed out after 5 minutes. Please try again.",
                    status_code=408,
                )

            server.handle_request()

        server.server_close()

        # Check for errors
        if OAuthCallbackHandler.error:
            raise GetAccessTokenError(
                message=f"OAuth error: {OAuthCallbackHandler.error}",
                status_code=400,
            )

        # Get the authorization code
        code = OAuthCallbackHandler.authorization_code
        callback_state = OAuthCallbackHandler.state

        # Decode state to get verifier
        if callback_state:
            state_data = decode_state(callback_state)
            verifier = state_data.get("verifier", verifier)

        # Exchange code for tokens
        return self._exchange_code_for_tokens(code, verifier)

    def _exchange_code_for_tokens(self, code: str, verifier: str) -> str:
        """Exchange authorization code for access and refresh tokens."""
        sync_client = _get_httpx_client()

        try:
            response = sync_client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": OAUTH_CLIENT_ID,
                    "client_secret": OAUTH_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": REDIRECT_URI,
                    "code_verifier": verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_data = response.json()
        except Exception as e:
            raise GetAccessTokenError(
                message=f"Failed to exchange code for tokens: {e}",
                status_code=400,
            )

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            raise GetAccessTokenError(
                message="No access token in response",
                status_code=400,
            )

        # Get user info
        email = self._get_user_email(access_token)

        # Discover project ID
        project_id = self._discover_project_id(access_token)

        # Save account
        account = {
            "email": email,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": time.time() + expires_in,
            "project_id": project_id,
        }
        self._save_account(account)

        print(  # noqa: T201
            f"\n{'='*60}\n"
            f"✓ Authentication successful!\n"
            f"  Account: {email}\n"
            f"  Project: {project_id}\n"
            f"{'='*60}\n",
            flush=True,
        )

        return access_token

    def _refresh_access_token(self, account: Dict[str, Any]) -> str:
        """Refresh the access token using refresh token."""
        refresh_token = account.get("refresh_token")

        if not refresh_token:
            raise GetRefreshTokenError(
                message="No refresh token available",
                status_code=401,
            )

        sync_client = _get_httpx_client()

        try:
            response = sync_client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": OAUTH_CLIENT_ID,
                    "client_secret": OAUTH_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_data = response.json()
        except Exception as e:
            raise GetRefreshTokenError(
                message=f"Failed to refresh token: {e}",
                status_code=401,
            )

        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            raise GetRefreshTokenError(
                message="No access token in refresh response",
                status_code=401,
            )

        # Update account
        account["access_token"] = access_token
        account["expires_at"] = time.time() + expires_in
        self._save_account(account)

        verbose_logger.info(f"Token refreshed for {account.get('email', 'unknown')}")

        return access_token

    def _get_user_email(self, access_token: str) -> str:
        """Get user email from Google userinfo endpoint."""
        try:
            sync_client = _get_httpx_client()
            response = sync_client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            user_info = response.json()
            return user_info.get("email", "unknown")
        except Exception as e:
            verbose_logger.warning(f"Failed to get user email: {e}")
            return "unknown"

    def _discover_project_id(self, access_token: str) -> str:
        """
        Discover project ID via loadCodeAssist API.

        Tries multiple Antigravity endpoints in order.
        Matches the exact request/response format from the reference implementation.
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "google-api-nodejs-client/9.15.1",
            "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            "Client-Metadata": json.dumps({
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }),
        }

        # Request body matches reference implementation
        request_body = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        }

        sync_client = _get_httpx_client()

        for endpoint in ANTIGRAVITY_ENDPOINTS:
            try:
                response = sync_client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers=headers,
                    json=request_body,
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Extract project ID - matching reference implementation
                    # Reference checks: cloudaicompanionProject (string or object with .id)
                    project_id = None

                    cloud_project = data.get("cloudaicompanionProject")
                    if isinstance(cloud_project, str) and cloud_project:
                        project_id = cloud_project
                    elif isinstance(cloud_project, dict) and cloud_project.get("id"):
                        project_id = cloud_project["id"]

                    # Fallback to other field names if present
                    if not project_id:
                        project_id = data.get("projectId") or data.get("managedProjectId")

                    if project_id:
                        verbose_logger.info(f"Discovered project ID: {project_id}")
                        return project_id
            except Exception as e:
                verbose_logger.debug(f"Project discovery failed for {endpoint}: {e}")
                continue

        # Fall back to default
        verbose_logger.info(f"Using default project ID: {DEFAULT_PROJECT_ID}")
        return DEFAULT_PROJECT_ID

    def _get_active_account(self) -> Optional[Dict[str, Any]]:
        """Get the active account from storage."""
        try:
            with open(self.accounts_file, "r") as f:
                data = json.load(f)
                accounts = data.get("accounts", [])
                if accounts:
                    return accounts[0]  # Return first account
        except (IOError, json.JSONDecodeError):
            pass
        return None

    def _save_account(self, account: Dict[str, Any]) -> None:
        """Save account to storage."""
        try:
            # Load existing accounts
            try:
                with open(self.accounts_file, "r") as f:
                    data = json.load(f)
            except (IOError, json.JSONDecodeError):
                data = {"version": 1, "accounts": []}

            accounts = data.get("accounts", [])

            # Update existing or add new
            email = account.get("email")
            found = False
            for i, acc in enumerate(accounts):
                if acc.get("email") == email:
                    accounts[i] = account
                    found = True
                    break

            if not found:
                accounts.insert(0, account)

            # Keep max 10 accounts
            data["accounts"] = accounts[:10]

            with open(self.accounts_file, "w") as f:
                json.dump(data, f, indent=2)

        except IOError as e:
            verbose_logger.error(f"Failed to save account: {e}")

    def _ensure_token_dir(self) -> None:
        """Ensure token directory exists with proper permissions."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, mode=0o700, exist_ok=True)

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            return result != 0
        except Exception:
            return True

    def logout(self) -> None:
        """Remove all stored accounts."""
        try:
            if os.path.exists(self.accounts_file):
                os.remove(self.accounts_file)
            print(f"Logged out. Tokens removed from {self.token_dir}")  # noqa: T201
        except IOError as e:
            verbose_logger.warning(f"Failed to remove accounts: {e}")

    def get_project_id(self) -> str:
        """Get the project ID from the active account."""
        account = self._get_active_account()
        if account:
            return account.get("project_id", DEFAULT_PROJECT_ID)
        return DEFAULT_PROJECT_ID


# =============================================================================
# Global Authenticator Instance
# =============================================================================

_authenticator: Optional[AntigravityAuthenticator] = None


def get_authenticator() -> AntigravityAuthenticator:
    """Get or create the global authenticator instance."""
    global _authenticator
    if _authenticator is None:
        _authenticator = AntigravityAuthenticator()
    return _authenticator


def get_antigravity_api_key() -> str:
    """
    Get an Antigravity API key (access token).

    This is the main entry point for obtaining authentication.
    If no valid token exists, it will initiate the OAuth flow.

    Returns:
        str: A valid OAuth access token.
    """
    return get_authenticator().get_api_key()


def get_antigravity_project_id() -> str:
    """
    Get the Antigravity project ID.

    Returns:
        str: The project ID from active account or default.
    """
    return get_authenticator().get_project_id()
