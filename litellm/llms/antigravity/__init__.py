# Antigravity API Provider for LiteLLM
# Google's Unified Gateway API for accessing multiple AI models (Claude, Gemini, GPT-OSS)
# through a single, consistent Gemini-style interface.

from litellm.llms.antigravity.chat.transformation import AntigravityConfig
from litellm.llms.antigravity.authenticator import (
    AntigravityAuthenticator,
    get_antigravity_api_key,
    get_antigravity_project_id,
    get_authenticator,
)
from litellm.llms.antigravity.common_utils import (
    AntigravityAuthError,
    GetAccessTokenError,
    GetRefreshTokenError,
    TokenExpiredError,
)

__all__ = [
    "AntigravityConfig",
    "AntigravityAuthenticator",
    "get_antigravity_api_key",
    "get_antigravity_project_id",
    "get_authenticator",
    "AntigravityAuthError",
    "GetAccessTokenError",
    "GetRefreshTokenError",
    "TokenExpiredError",
]
