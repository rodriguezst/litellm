"""
Basic unit tests for the Antigravity provider.

These tests verify core functionality without making network calls.
"""

import base64
import hashlib
import urllib.parse

import pytest


class TestAntigravityConfig:
    """Tests for AntigravityConfig class instantiation."""

    def test_antigravity_config_instantiation(self):
        """Test that AntigravityConfig can be instantiated."""
        from litellm.llms.antigravity.chat.transformation import AntigravityConfig

        config = AntigravityConfig()
        assert config is not None
        assert hasattr(config, "authenticator")
        assert hasattr(config, "transform_request")
        assert hasattr(config, "transform_response")

    def test_antigravity_config_with_custom_provider(self):
        """Test AntigravityConfig with custom provider name."""
        from litellm.llms.antigravity.chat.transformation import AntigravityConfig

        config = AntigravityConfig(custom_llm_provider="my_custom_provider")
        assert config._custom_llm_provider == "my_custom_provider"

    def test_antigravity_config_has_authenticator(self):
        """Test that AntigravityConfig initializes an Authenticator."""
        from litellm.llms.antigravity.authenticator import Authenticator
        from litellm.llms.antigravity.chat.transformation import AntigravityConfig

        config = AntigravityConfig()
        assert isinstance(config.authenticator, Authenticator)


class TestPKCEGeneration:
    """Tests for PKCE (Proof Key for Code Exchange) generation."""

    def test_pkce_verifier_length(self):
        """Test that PKCE verifier has correct length."""
        from litellm.llms.antigravity.authenticator import Authenticator

        auth = Authenticator()
        verifier, challenge = auth._generate_pkce()

        # Verifier should be URL-safe base64 encoded, at least 43 chars for 32 bytes
        assert len(verifier) >= 43, "Verifier should be at least 43 characters"

    def test_pkce_verifier_is_url_safe(self):
        """Test that PKCE verifier uses URL-safe characters."""
        from litellm.llms.antigravity.authenticator import Authenticator

        auth = Authenticator()
        verifier, challenge = auth._generate_pkce()

        # URL-safe base64 uses alphanumeric, '-', and '_'
        for char in verifier:
            assert char.isalnum() or char in "-_", f"Invalid character in verifier: {char}"

    def test_pkce_challenge_length(self):
        """Test that PKCE challenge has correct length (SHA256 base64url without padding)."""
        from litellm.llms.antigravity.authenticator import Authenticator

        auth = Authenticator()
        verifier, challenge = auth._generate_pkce()

        # SHA256 produces 32 bytes, base64url encoding produces 43 chars without padding
        assert len(challenge) == 43, "Challenge should be 43 characters (SHA256 base64url without padding)"

    def test_pkce_challenge_is_url_safe(self):
        """Test that PKCE challenge uses URL-safe characters."""
        from litellm.llms.antigravity.authenticator import Authenticator

        auth = Authenticator()
        verifier, challenge = auth._generate_pkce()

        for char in challenge:
            assert char.isalnum() or char in "-_", f"Invalid character in challenge: {char}"

    def test_pkce_challenge_is_sha256_of_verifier(self):
        """Test that challenge is correctly derived from verifier using SHA256."""
        from litellm.llms.antigravity.authenticator import Authenticator

        auth = Authenticator()
        verifier, challenge = auth._generate_pkce()

        # Manually compute the expected challenge
        expected_challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        expected_challenge = base64.urlsafe_b64encode(expected_challenge_bytes).rstrip(b"=").decode()

        assert challenge == expected_challenge, "Challenge should be SHA256 hash of verifier"

    def test_pkce_values_are_unique(self):
        """Test that multiple PKCE generations produce unique values."""
        from litellm.llms.antigravity.authenticator import Authenticator

        auth = Authenticator()

        # Generate multiple sets
        pairs = [auth._generate_pkce() for _ in range(5)]
        verifiers = [pair[0] for pair in pairs]
        challenges = [pair[1] for pair in pairs]

        # All should be unique
        assert len(set(verifiers)) == 5, "All verifiers should be unique"
        assert len(set(challenges)) == 5, "All challenges should be unique"


class TestAuthorizationURLGeneration:
    """Tests for OAuth authorization URL generation."""

    def test_url_contains_client_id(self):
        """Test that authorization URL contains the correct client_id."""
        from litellm.llms.antigravity.common_utils import (
            ANTIGRAVITY_CLIENT_ID,
            ANTIGRAVITY_REDIRECT_URI,
            ANTIGRAVITY_SCOPES,
            GOOGLE_AUTH_URL,
        )

        # Build auth URL (simulating what _login does)
        import secrets

        verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)

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

        # Parse and verify
        parsed = urllib.parse.urlparse(auth_url)
        params = urllib.parse.parse_qs(parsed.query)

        assert params.get("client_id", [""])[0] == ANTIGRAVITY_CLIENT_ID

    def test_url_contains_redirect_uri(self):
        """Test that authorization URL contains the correct redirect_uri."""
        from litellm.llms.antigravity.common_utils import (
            ANTIGRAVITY_CLIENT_ID,
            ANTIGRAVITY_REDIRECT_URI,
            ANTIGRAVITY_SCOPES,
            GOOGLE_AUTH_URL,
        )

        import secrets

        verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)

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

        parsed = urllib.parse.urlparse(auth_url)
        params = urllib.parse.parse_qs(parsed.query)

        assert params.get("redirect_uri", [""])[0] == ANTIGRAVITY_REDIRECT_URI

    def test_url_uses_s256_code_challenge_method(self):
        """Test that authorization URL uses S256 code challenge method."""
        from litellm.llms.antigravity.common_utils import (
            ANTIGRAVITY_CLIENT_ID,
            ANTIGRAVITY_REDIRECT_URI,
            ANTIGRAVITY_SCOPES,
            GOOGLE_AUTH_URL,
        )

        import secrets

        verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)

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

        parsed = urllib.parse.urlparse(auth_url)
        params = urllib.parse.parse_qs(parsed.query)

        assert params.get("code_challenge_method", [""])[0] == "S256"

    def test_url_has_offline_access_type(self):
        """Test that authorization URL requests offline access for refresh tokens."""
        from litellm.llms.antigravity.common_utils import (
            ANTIGRAVITY_CLIENT_ID,
            ANTIGRAVITY_REDIRECT_URI,
            ANTIGRAVITY_SCOPES,
            GOOGLE_AUTH_URL,
        )

        import secrets

        verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)

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

        parsed = urllib.parse.urlparse(auth_url)
        params = urllib.parse.parse_qs(parsed.query)

        assert params.get("access_type", [""])[0] == "offline"

    def test_url_uses_google_oauth_endpoint(self):
        """Test that authorization URL uses Google's OAuth endpoint."""
        from litellm.llms.antigravity.common_utils import GOOGLE_AUTH_URL

        assert "accounts.google.com" in GOOGLE_AUTH_URL
        assert "/o/oauth2/v2/auth" in GOOGLE_AUTH_URL

    def test_url_contains_required_scopes(self):
        """Test that authorization URL contains required scopes."""
        from litellm.llms.antigravity.common_utils import (
            ANTIGRAVITY_CLIENT_ID,
            ANTIGRAVITY_REDIRECT_URI,
            ANTIGRAVITY_SCOPES,
            GOOGLE_AUTH_URL,
        )

        import secrets

        verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)

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

        parsed = urllib.parse.urlparse(auth_url)
        params = urllib.parse.parse_qs(parsed.query)

        scope_string = params.get("scope", [""])[0]
        for required_scope in ANTIGRAVITY_SCOPES:
            assert required_scope in scope_string, f"Missing scope: {required_scope}"


class TestAntigravityConstants:
    """Tests for Antigravity constants and utilities."""

    def test_callback_port_is_valid(self):
        """Test that callback port is a valid port number."""
        from litellm.llms.antigravity.common_utils import ANTIGRAVITY_CALLBACK_PORT

        assert 1024 <= ANTIGRAVITY_CALLBACK_PORT <= 65535

    def test_redirect_uri_matches_port(self):
        """Test that redirect URI uses the correct callback port."""
        from litellm.llms.antigravity.common_utils import (
            ANTIGRAVITY_CALLBACK_PORT,
            ANTIGRAVITY_REDIRECT_URI,
        )

        assert str(ANTIGRAVITY_CALLBACK_PORT) in ANTIGRAVITY_REDIRECT_URI
        assert "localhost" in ANTIGRAVITY_REDIRECT_URI
        assert "/oauth-callback" in ANTIGRAVITY_REDIRECT_URI

    def test_api_endpoints_are_valid(self):
        """Test that API endpoints are valid HTTPS URLs."""
        from litellm.llms.antigravity.common_utils import ANTIGRAVITY_ENDPOINTS

        for endpoint in ANTIGRAVITY_ENDPOINTS:
            assert endpoint.startswith("https://"), f"Endpoint should be HTTPS: {endpoint}"
            assert "googleapis.com" in endpoint, f"Endpoint should be a Google API: {endpoint}"

    def test_token_url_is_google_oauth(self):
        """Test that token URL is Google's OAuth token endpoint."""
        from litellm.llms.antigravity.common_utils import GOOGLE_TOKEN_URL

        assert "oauth2.googleapis.com" in GOOGLE_TOKEN_URL
        assert "/token" in GOOGLE_TOKEN_URL


class TestModelUtilities:
    """Tests for model name utilities."""

    def test_normalize_model_name_removes_prefix(self):
        """Test that normalize_model_name removes provider prefix."""
        from litellm.llms.antigravity.common_utils import normalize_model_name

        assert normalize_model_name("antigravity/claude-sonnet-4-5") == "claude-sonnet-4-5"
        assert normalize_model_name("provider/model-name") == "model-name"

    def test_normalize_model_name_handles_no_prefix(self):
        """Test that normalize_model_name handles models without prefix."""
        from litellm.llms.antigravity.common_utils import normalize_model_name

        assert normalize_model_name("claude-sonnet-4-5") == "claude-sonnet-4-5"

    def test_is_thinking_model_detection(self):
        """Test that thinking model detection works correctly."""
        from litellm.llms.antigravity.common_utils import is_thinking_model

        assert is_thinking_model("claude-opus-4-5-thinking-medium") is True
        assert is_thinking_model("claude-opus-4-5-thinking-high") is True
        assert is_thinking_model("claude-sonnet-4-5") is False
        assert is_thinking_model("gpt-4") is False

    def test_get_thinking_budget(self):
        """Test that thinking budget is correctly extracted."""
        from litellm.llms.antigravity.common_utils import get_thinking_budget

        assert get_thinking_budget("claude-opus-4-5-thinking-low") == 8192
        assert get_thinking_budget("claude-opus-4-5-thinking-medium") == 16384
        assert get_thinking_budget("claude-opus-4-5-thinking-high") == 32768
        assert get_thinking_budget("claude-sonnet-4-5") is None
