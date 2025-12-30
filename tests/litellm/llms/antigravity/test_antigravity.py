"""
Unit tests for Antigravity API provider in LiteLLM.

Antigravity is Google's Unified Gateway API for accessing multiple AI models
(Claude, Gemini, GPT-OSS) through a single, consistent Gemini-style interface.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
import httpx

from litellm.llms.antigravity.chat.transformation import (
    AntigravityConfig,
    AntigravityError,
    AntigravityStreamingIterator,
    AntigravityAsyncStreamingIterator,
)
from litellm.types.utils import ModelResponse, Choices, Message, Usage


class TestAntigravityConfig:
    """Test suite for AntigravityConfig class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_ui_friendly_name(self):
        """Test the UI friendly name."""
        assert self.config.ui_friendly_name() == "Antigravity (Google Unified Gateway)"

    def test_is_thinking_model(self):
        """Test thinking model detection."""
        # Thinking models
        assert self.config.is_thinking_model("claude-sonnet-4-5-thinking-low") is True
        assert self.config.is_thinking_model("claude-sonnet-4-5-thinking-medium") is True
        assert self.config.is_thinking_model("claude-sonnet-4-5-thinking-high") is True
        assert self.config.is_thinking_model("claude-opus-4-5-thinking-high") is True

        # Non-thinking models
        assert self.config.is_thinking_model("claude-sonnet-4-5") is False
        assert self.config.is_thinking_model("gemini-3-pro-low") is False
        assert self.config.is_thinking_model("gpt-oss-120b-medium") is False

    def test_get_thinking_budget_from_model(self):
        """Test thinking budget extraction from model name."""
        assert self.config.get_thinking_budget_from_model("claude-sonnet-4-5-thinking-low") == 8192
        assert self.config.get_thinking_budget_from_model("claude-sonnet-4-5-thinking-medium") == 16384
        assert self.config.get_thinking_budget_from_model("claude-sonnet-4-5-thinking-high") == 32768
        assert self.config.get_thinking_budget_from_model("claude-sonnet-4-5") is None

    def test_is_claude_model(self):
        """Test Claude model detection."""
        assert self.config._is_claude_model("claude-sonnet-4-5") is True
        assert self.config._is_claude_model("Claude-Opus-4-5") is True
        assert self.config._is_claude_model("gemini-3-pro-low") is False
        assert self.config._is_claude_model("gpt-oss-120b-medium") is False

    def test_is_gemini_model(self):
        """Test Gemini model detection."""
        assert self.config._is_gemini_model("gemini-3-pro-low") is True
        assert self.config._is_gemini_model("Gemini-3-flash") is True
        assert self.config._is_gemini_model("claude-sonnet-4-5") is False
        assert self.config._is_gemini_model("gpt-oss-120b-medium") is False

    def test_get_supported_openai_params(self):
        """Test supported parameters list."""
        # Non-thinking model
        params = self.config.get_supported_openai_params("claude-sonnet-4-5")
        assert "temperature" in params
        assert "max_tokens" in params
        assert "stream" in params
        assert "tools" in params
        assert "thinking" not in params
        assert "reasoning_effort" not in params

        # Thinking model
        thinking_params = self.config.get_supported_openai_params("claude-sonnet-4-5-thinking-high")
        assert "thinking" in thinking_params
        assert "reasoning_effort" in thinking_params

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping."""
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95,
            "stop": ["END"],
            "n": 2,
            "stream": True,
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, "claude-sonnet-4-5", False
        )

        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 1000
        assert result["top_p"] == 0.95
        assert result["stop_sequences"] == ["END"]
        assert result["candidate_count"] == 2
        assert result["stream"] is True

    def test_map_openai_params_stop_string(self):
        """Test stop parameter mapping with string input."""
        non_default_params = {"stop": "STOP"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, "claude-sonnet-4-5", False
        )

        assert result["stop_sequences"] == ["STOP"]

    def test_get_complete_url_non_streaming(self):
        """Test URL generation for non-streaming requests."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="claude-sonnet-4-5",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:generateContent"

    def test_get_complete_url_streaming(self):
        """Test URL generation for streaming requests."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="claude-sonnet-4-5",
            optional_params={},
            litellm_params={},
            stream=True,
        )
        assert url == "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse"

    def test_get_complete_url_custom_base(self):
        """Test URL generation with custom API base."""
        url = self.config.get_complete_url(
            api_base="https://custom-api.example.com",
            api_key="test-key",
            model="claude-sonnet-4-5",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "https://custom-api.example.com/v1internal:generateContent"


class TestAntigravityMessageTransformation:
    """Test suite for message transformation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_transform_simple_messages(self):
        """Test simple message transformation."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        result = self.config._transform_messages_to_gemini_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "Hello, how are you?"

    def test_transform_assistant_message(self):
        """Test assistant message transformation (role mapping)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = self.config._transform_messages_to_gemini_format(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"  # assistant -> model

    def test_transform_system_message_skipped(self):
        """Test that system messages are skipped (they go to systemInstruction)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        result = self.config._transform_messages_to_gemini_format(messages)

        # System message should be skipped
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_get_system_instruction(self):
        """Test system instruction extraction."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        result = self.config._get_system_instruction(messages)

        assert result is not None
        assert result["parts"][0]["text"] == "You are a helpful assistant."

    def test_get_system_instruction_none(self):
        """Test system instruction extraction when none exists."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        result = self.config._get_system_instruction(messages)

        assert result is None

    def test_transform_tool_call_message(self):
        """Test tool call message transformation."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}'
                        }
                    }
                ]
            }
        ]

        result = self.config._transform_messages_to_gemini_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert "functionCall" in result[0]["parts"][0]
        assert result[0]["parts"][0]["functionCall"]["name"] == "get_weather"
        assert result[0]["parts"][0]["functionCall"]["args"] == {"location": "Paris"}

    def test_transform_tool_result_message(self):
        """Test tool result message transformation."""
        messages = [
            {
                "role": "tool",
                "content": "Sunny, 22°C",
                "tool_call_id": "call_123",
                "name": "get_weather",
            }
        ]

        result = self.config._transform_messages_to_gemini_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"  # Tool results are user role
        assert "functionResponse" in result[0]["parts"][0]
        assert result[0]["parts"][0]["functionResponse"]["name"] == "get_weather"


class TestAntigravityToolTransformation:
    """Test suite for tool transformation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_transform_simple_tool(self):
        """Test simple tool transformation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        result = self.config._transform_tools_to_gemini_format(tools)

        assert result is not None
        assert len(result) == 1
        assert "functionDeclarations" in result[0]
        assert result[0]["functionDeclarations"][0]["name"] == "get_weather"
        assert result[0]["functionDeclarations"][0]["description"] == "Get weather for a location"

    def test_transform_empty_tools(self):
        """Test empty tools list."""
        result = self.config._transform_tools_to_gemini_format([])
        assert result is None

    def test_tool_name_slash_replacement(self):
        """Test that slashes in tool names are replaced with underscores."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp/query",
                    "description": "Query MCP",
                }
            }
        ]

        result = self.config._transform_tools_to_gemini_format(tools)

        # Slashes should be replaced with underscores
        assert result[0]["functionDeclarations"][0]["name"] == "mcp_query"


class TestAntigravitySchemaCleanup:
    """Test suite for JSON schema cleanup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_clean_const_to_enum(self):
        """Test that 'const' is converted to 'enum'."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"const": "active"}
            }
        }

        result = self.config._clean_schema_for_antigravity(schema)

        assert "enum" in result["properties"]["status"]
        assert result["properties"]["status"]["enum"] == ["active"]
        assert "const" not in result["properties"]["status"]

    def test_clean_unsupported_fields(self):
        """Test removal of unsupported fields."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "test-schema",
            "$ref": "#/definitions/Something",
            "$defs": {"Something": {"type": "string"}},
            "default": "hello",
            "examples": ["hello", "world"],
            "type": "string"
        }

        result = self.config._clean_schema_for_antigravity(schema)

        assert "$schema" not in result
        assert "$id" not in result
        assert "$ref" not in result
        assert "$defs" not in result
        assert "default" not in result
        assert "examples" not in result
        assert result["type"] == "string"

    def test_clean_anyof_renamed(self):
        """Test that 'anyOf' is renamed to 'any_of'."""
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        }

        result = self.config._clean_schema_for_antigravity(schema)

        assert "any_of" in result
        assert "anyOf" not in result


class TestAntigravityGenerationConfig:
    """Test suite for generation config building."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_basic_generation_config(self):
        """Test basic generation config building."""
        optional_params = {
            "temperature": 0.7,
            "max_output_tokens": 1000,
            "top_p": 0.95,
            "stop_sequences": ["END"]
        }

        result = self.config._get_generation_config("claude-sonnet-4-5", optional_params)

        assert result["temperature"] == 0.7
        assert result["maxOutputTokens"] == 1000
        assert result["topP"] == 0.95
        assert result["stopSequences"] == ["END"]

    def test_thinking_model_config(self):
        """Test generation config for thinking models."""
        optional_params = {}

        result = self.config._get_generation_config("claude-sonnet-4-5-thinking-high", optional_params)

        assert "thinkingConfig" in result
        assert result["thinkingConfig"]["thinkingBudget"] == 32768
        assert result["thinkingConfig"]["includeThoughts"] is True


class TestAntigravityRequestTransform:
    """Test suite for request transformation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_basic_request_transform(self):
        """Test basic request transformation."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        optional_params = {"temperature": 0.7}
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="claude-sonnet-4-5",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert result["model"] == "claude-sonnet-4-5"
        assert result["project"] == "rising-fact-p41fc"  # default project
        assert result["userAgent"] == "litellm-antigravity"
        assert "requestId" in result
        assert "request" in result
        assert "contents" in result["request"]
        assert "generationConfig" in result["request"]
        assert result["request"]["generationConfig"]["temperature"] == 0.7

    def test_request_with_system_instruction(self):
        """Test request with system instruction."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]

        result = self.config.transform_request(
            model="claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert "systemInstruction" in result["request"]
        assert result["request"]["systemInstruction"]["parts"][0]["text"] == "You are helpful."

    def test_request_with_custom_project(self):
        """Test request with custom project ID."""
        result = self.config.transform_request(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"project": "custom-project-123"},
            headers={},
        )

        assert result["project"] == "custom-project-123"


class TestAntigravityResponseTransform:
    """Test suite for response transformation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_basic_response_transform(self):
        """Test basic response transformation."""
        # Create mock response
        response_data = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello! I'm doing well."}]
                        },
                        "finishReason": "STOP"
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 18
                },
                "modelVersion": "claude-sonnet-4-5",
                "responseId": "msg_123"
            }
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data

        model_response = ModelResponse()

        result = self.config.transform_response(
            model="claude-sonnet-4-5",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello! I'm doing well."
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 8
        assert result.usage.total_tokens == 18
        assert result.model == "claude-sonnet-4-5"

    def test_response_with_tool_call(self):
        """Test response transformation with tool call."""
        response_data = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris"},
                                    "id": "call_123"
                                }
                            }]
                        },
                        "finishReason": "OTHER"
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15
                }
            }
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data

        model_response = ModelResponse()

        result = self.config.transform_response(
            model="claude-sonnet-4-5",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content is None
        assert "tool_calls" in result.choices[0].message.model_dump()

    def test_response_error_handling(self):
        """Test error response handling."""
        response_data = {
            "error": {
                "code": 400,
                "message": "Invalid request",
                "status": "INVALID_ARGUMENT"
            }
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data

        model_response = ModelResponse()

        with pytest.raises(AntigravityError) as exc_info:
            self.config.transform_response(
                model="claude-sonnet-4-5",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=MagicMock(),
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid request" in exc_info.value.message


class TestAntigravityValidateEnvironment:
    """Test suite for environment validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_validate_with_api_key(self):
        """Test validation with API key provided."""
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model="claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        assert result["Authorization"] == "Bearer test-api-key"
        assert result["Content-Type"] == "application/json"
        assert "User-Agent" in result
        assert "X-Goog-Api-Client" in result
        assert "Client-Metadata" in result

    def test_validate_without_api_key(self):
        """Test validation without API key raises error."""
        headers = {}

        with pytest.raises(ValueError) as exc_info:
            self.config.validate_environment(
                headers=headers,
                model="claude-sonnet-4-5",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

        assert "ANTIGRAVITY_API_KEY is not set" in str(exc_info.value)


class TestAntigravityFinishReasonMapping:
    """Test suite for finish reason mapping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AntigravityConfig()

    def test_finish_reason_mapping(self):
        """Test all finish reason mappings."""
        assert self.config._map_finish_reason("STOP") == "stop"
        assert self.config._map_finish_reason("MAX_TOKENS") == "length"
        assert self.config._map_finish_reason("SAFETY") == "content_filter"
        assert self.config._map_finish_reason("RECITATION") == "content_filter"
        assert self.config._map_finish_reason("OTHER") == "stop"
        assert self.config._map_finish_reason("BLOCKLIST") == "content_filter"
        assert self.config._map_finish_reason("UNKNOWN") == "stop"  # default


class TestAntigravityError:
    """Test suite for AntigravityError."""

    def test_error_creation(self):
        """Test error creation."""
        error = AntigravityError(
            status_code=429,
            message="Rate limit exceeded",
            headers={"Retry-After": "30"},
        )

        assert error.status_code == 429
        assert error.message == "Rate limit exceeded"
        assert error.headers == {"Retry-After": "30"}


class TestAntigravityStreamingIterator:
    """Test suite for streaming iterator."""

    def test_process_chunk(self):
        """Test SSE chunk processing."""
        iterator = AntigravityStreamingIterator(iter([]))

        chunk = 'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}]}}'

        result = iterator._process_chunk(chunk)

        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_process_empty_chunk(self):
        """Test empty chunk handling."""
        iterator = AntigravityStreamingIterator(iter([]))

        result = iterator._process_chunk("")

        assert result["choices"][0]["delta"] == {}

    def test_process_chunk_with_finish_reason(self):
        """Test chunk with finish reason."""
        iterator = AntigravityStreamingIterator(iter([]))

        chunk = 'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP"}]}}'

        result = iterator._process_chunk(chunk)

        assert result["choices"][0]["finish_reason"] == "stop"


class TestAntigravityAuthenticator:
    """Test suite for Antigravity OAuth authenticator functions."""

    def test_generate_pkce_pair(self):
        """Test PKCE code verifier and challenge generation."""
        from litellm.llms.antigravity.authenticator import generate_pkce_pair

        verifier, challenge = generate_pkce_pair()

        # Verifier should be 43-128 characters (URL-safe base64)
        assert len(verifier) >= 43
        assert len(verifier) <= 128
        # Challenge should be 43 characters (SHA256 -> base64url without padding)
        assert len(challenge) == 43
        # Both should be URL-safe
        assert "/" not in challenge
        assert "+" not in challenge

    def test_pkce_pair_uniqueness(self):
        """Test that PKCE pairs are unique each time."""
        from litellm.llms.antigravity.authenticator import generate_pkce_pair

        verifier1, challenge1 = generate_pkce_pair()
        verifier2, challenge2 = generate_pkce_pair()

        assert verifier1 != verifier2
        assert challenge1 != challenge2

    def test_encode_decode_state(self):
        """Test OAuth state encoding and decoding."""
        from litellm.llms.antigravity.authenticator import encode_state, decode_state

        verifier = "test_verifier_12345"
        project_id = "my-gcp-project"

        state = encode_state(verifier, project_id)
        decoded = decode_state(state)

        assert decoded["verifier"] == verifier
        assert decoded["projectId"] == project_id

    def test_decode_state_empty(self):
        """Test decoding state with empty project ID."""
        from litellm.llms.antigravity.authenticator import encode_state, decode_state

        verifier = "test_verifier"
        state = encode_state(verifier, "")
        decoded = decode_state(state)

        assert decoded["verifier"] == verifier
        assert decoded["projectId"] == ""

    def test_decode_state_invalid(self):
        """Test decoding invalid state returns empty values."""
        from litellm.llms.antigravity.authenticator import decode_state

        # Invalid base64
        decoded = decode_state("not-valid-base64!!!")

        assert decoded["verifier"] == ""
        assert decoded["projectId"] == ""

    def test_oauth_constants(self):
        """Test that OAuth constants are correctly configured."""
        from litellm.llms.antigravity.authenticator import (
            OAUTH_CLIENT_ID,
            REDIRECT_PORT,
            REDIRECT_URI,
            OAUTH_SCOPES,
            DEFAULT_PROJECT_ID,
            GOOGLE_AUTH_URL,
            GOOGLE_TOKEN_URL,
        )

        # Client ID format
        assert OAUTH_CLIENT_ID.endswith(".apps.googleusercontent.com")

        # Redirect configuration
        assert REDIRECT_PORT == 51121
        assert REDIRECT_URI == "http://localhost:51121/oauth-callback"

        # Scopes
        assert len(OAUTH_SCOPES) == 5
        assert "https://www.googleapis.com/auth/cloud-platform" in OAUTH_SCOPES
        assert "https://www.googleapis.com/auth/userinfo.email" in OAUTH_SCOPES

        # URLs
        assert GOOGLE_AUTH_URL == "https://accounts.google.com/o/oauth2/v2/auth"
        assert GOOGLE_TOKEN_URL == "https://oauth2.googleapis.com/token"

        # Default project
        assert DEFAULT_PROJECT_ID == "rising-fact-p41fc"

    def test_authenticator_token_dir(self):
        """Test authenticator creates token directory path."""
        import os
        from unittest.mock import patch

        # Mock the file system operations
        with patch.object(os, 'makedirs') as mock_makedirs:
            with patch.object(os.path, 'exists', return_value=True):
                from litellm.llms.antigravity.authenticator import AntigravityAuthenticator

                auth = AntigravityAuthenticator()

                assert "antigravity" in auth.token_dir
                assert auth.accounts_file.endswith("accounts.json")
