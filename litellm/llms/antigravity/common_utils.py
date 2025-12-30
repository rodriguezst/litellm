"""
Common utilities and error classes for Antigravity provider.
"""

from typing import Optional, Union
import httpx


class AntigravityAuthError(Exception):
    """Base class for Antigravity authentication errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 401,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.headers = headers
        super().__init__(self.message)


class GetAccessTokenError(AntigravityAuthError):
    """Error when unable to obtain an access token."""
    pass


class GetRefreshTokenError(AntigravityAuthError):
    """Error when unable to refresh the access token."""
    pass


class TokenExpiredError(AntigravityAuthError):
    """Error when the token has expired."""
    pass


class GetDeviceCodeError(AntigravityAuthError):
    """Error when unable to obtain a device code."""
    pass


# Default headers for Antigravity API requests
def get_antigravity_headers(access_token: str) -> dict:
    """
    Get standard headers for Antigravity API requests.

    Args:
        access_token: The OAuth access token.

    Returns:
        dict: Headers for API requests.
    """
    import json

    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "litellm-antigravity/1.0",
        "X-Goog-Api-Client": "google-cloud-sdk litellm/0.1",
        "Client-Metadata": json.dumps({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI"
        }),
    }
