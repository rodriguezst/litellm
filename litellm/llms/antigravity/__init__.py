"""
Antigravity provider for LiteLLM.

Antigravity is Google's unified AI gateway providing access to Claude, Gemini,
and other models through a single API.
"""

from .authenticator import Authenticator
from .common_utils import (
    ANTIGRAVITY_API_BASE,
    ANTIGRAVITY_DEFAULT_PROJECT,
    ANTIGRAVITY_ENDPOINTS,
    ANTIGRAVITY_LOAD_ENDPOINTS,
    ANTHROPIC_BETA_HEADER,
    AntigravityError,
    AuthenticationError,
    CallbackServerError,
    ProjectDiscoveryError,
    TokenExpiredError,
    TokenRefreshError,
    get_antigravity_default_headers,
    get_streaming_headers,
)
from .chat.transformation import AntigravityConfig

__all__ = [
    "Authenticator",
    "AntigravityConfig",
    "AntigravityError",
    "AuthenticationError",
    "TokenRefreshError",
    "TokenExpiredError",
    "ProjectDiscoveryError",
    "CallbackServerError",
    "ANTIGRAVITY_API_BASE",
    "ANTIGRAVITY_DEFAULT_PROJECT",
    "ANTIGRAVITY_ENDPOINTS",
    "ANTIGRAVITY_LOAD_ENDPOINTS",
    "ANTHROPIC_BETA_HEADER",
    "get_antigravity_default_headers",
    "get_streaming_headers",
]
