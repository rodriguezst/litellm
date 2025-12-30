"""
Constants and utilities for Antigravity integration.

Antigravity is Google's unified AI gateway providing access to Claude, Gemini, and other models.
"""
from typing import Optional, Union
from uuid import uuid4

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# OAuth Configuration
ANTIGRAVITY_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
ANTIGRAVITY_CLIENT_SECRET = "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"
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_CALLBACK_PORT = 51121

ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# OAuth URLs
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"

# API Endpoints (in fallback order for requests)
ANTIGRAVITY_ENDPOINTS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]

# Endpoints for project discovery (production first per reference)
ANTIGRAVITY_LOAD_ENDPOINTS = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
]

# Default API endpoint
ANTIGRAVITY_API_BASE = ANTIGRAVITY_ENDPOINTS[0]

# Default project ID (used when project discovery fails)
ANTIGRAVITY_DEFAULT_PROJECT = "rising-fact-p41fc"

# API Paths
ANTIGRAVITY_GENERATE_PATH = "/v1internal:generateContent"
ANTIGRAVITY_STREAM_PATH = "/v1internal:streamGenerateContent"

# Model mapping from common names to Antigravity model names
MODEL_ALIASES = {
    "claude-3.5-sonnet": "claude-sonnet-4-5",
    "claude-sonnet-3.5": "claude-sonnet-4-5",
    "claude-sonnet": "claude-sonnet-4-5",
    "claude-opus": "claude-opus-4-5-thinking-medium",
}

# Thinking model budget configurations
THINKING_TIER_BUDGETS = {
    "low": 8192,
    "medium": 16384,
    "high": 32768,
}

# Claude thinking models require maxOutputTokens >= thinkingBudget
CLAUDE_THINKING_MAX_OUTPUT_TOKENS = 64000

# Claude-specific beta header for interleaved thinking
ANTHROPIC_BETA_HEADER = "interleaved-thinking-2025-05-14"

# Claude tool hardening system instruction (prevents parameter hallucination)
CLAUDE_TOOL_SYSTEM_INSTRUCTION = """
When using tools, you MUST provide all required parameters exactly as specified.
Do not invent or hallucinate parameter values. If a parameter value is unknown,
ask the user for clarification instead of guessing.
"""

# Unsupported JSON Schema keywords (must be stripped before sending to API)
UNSUPPORTED_SCHEMA_KEYWORDS = [
    "minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum",
    "pattern", "minItems", "maxItems", "format", "default", "examples",
    "$schema", "$defs", "definitions", "const", "$ref", "additionalProperties",
    "propertyNames", "title", "$id", "$comment",
]


class AntigravityError(BaseLLMException):
    """Base exception for Antigravity API errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class AuthenticationError(AntigravityError):
    """Error during OAuth authentication flow."""
    pass


class TokenRefreshError(AntigravityError):
    """Error refreshing access token."""
    pass


class TokenExpiredError(AntigravityError):
    """Access token has expired."""
    pass


class ProjectDiscoveryError(AntigravityError):
    """Error discovering project ID."""
    pass


class CallbackServerError(AntigravityError):
    """Error with local OAuth callback server."""
    pass


def get_antigravity_default_headers(access_token: str) -> dict:
    """
    Get default headers for Antigravity API requests.

    Based on the reference implementation headers.
    """
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "antigravity/1.11.5 linux/amd64",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
        "X-Request-Id": str(uuid4()),
    }


def get_streaming_headers(access_token: str) -> dict:
    """
    Get headers for streaming API requests.
    """
    headers = get_antigravity_default_headers(access_token)
    headers["Accept"] = "text/event-stream"
    return headers


def is_thinking_model(model: str) -> bool:
    """Check if the model is a thinking model (has thinking budget)."""
    return "thinking" in model.lower()


def get_thinking_budget(model: str) -> Optional[int]:
    """Get the thinking budget for a thinking model."""
    model_lower = model.lower()
    for tier, budget in THINKING_TIER_BUDGETS.items():
        if f"thinking-{tier}" in model_lower:
            return budget
    return None


def normalize_model_name(model: str) -> str:
    """Normalize model name using aliases."""
    # Remove provider prefix if present
    if "/" in model:
        model = model.split("/", 1)[1]

    # Check for aliases
    return MODEL_ALIASES.get(model, model)


def clean_json_schema(schema: dict) -> dict:
    """
    Clean JSON schema by removing unsupported keywords.

    The Antigravity API doesn't support certain JSON Schema features,
    so we need to strip them to avoid 400 errors.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    for key, value in schema.items():
        if key in UNSUPPORTED_SCHEMA_KEYWORDS:
            continue

        if isinstance(value, dict):
            cleaned[key] = clean_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                clean_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value

    return cleaned
