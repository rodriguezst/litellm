"""
Antigravity API Provider for LiteLLM

Google's Unified Gateway API for accessing multiple AI models (Claude, Gemini, GPT-OSS)
through a single, consistent Gemini-style interface.

Key Features:
- OAuth 2.0 authentication with Google Cloud Platform
- Single API format for all models using Gemini-style `contents` array
- Multi-account load balancing with automatic failover
- Real-time SSE streaming support

Endpoints:
- Daily (Sandbox): https://daily-cloudcode-pa.sandbox.googleapis.com
- Production: https://cloudcode-pa.googleapis.com

Available Models:
- Claude: claude-sonnet-4-5, claude-sonnet-4-5-thinking-{budget}, claude-opus-4-5-thinking-{budget}
- Gemini: gemini-3-pro-low, gemini-3-pro-high, gemini-3-flash
- GPT-OSS: gpt-oss-120b-medium
"""

import json
import time
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx

import litellm
from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class AntigravityError(BaseLLMException):
    """Exception class for Antigravity API errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
            request=request,
            response=response,
            body=body,
        )


class AntigravityConfig(BaseConfig):
    """
    Configuration for Antigravity API - Google's Unified Gateway API.

    Reference: https://github.com/NoeFabris/opencode-antigravity-auth

    Supported Models:
    - Claude: claude-sonnet-4-5, claude-sonnet-4-5-thinking-{low,medium,high}
    - Gemini: gemini-3-pro-low, gemini-3-pro-high, gemini-3-flash
    - GPT-OSS: gpt-oss-120b-medium

    Environment Variables:
    - ANTIGRAVITY_API_KEY: OAuth access token (Bearer token)
    - ANTIGRAVITY_REFRESH_TOKEN: OAuth refresh token for auto-refresh
    - ANTIGRAVITY_PROJECT_ID: GCP project ID (default: rising-fact-p41fc)
    - ANTIGRAVITY_API_BASE: API base URL (default: daily sandbox)

    Parameters:
    - temperature (float): Controls randomness in token selection (0.0-2.0)
    - max_output_tokens (int): Maximum tokens in the output
    - top_p (float): Top-p sampling parameter (0.0-1.0)
    - top_k (int): Top-k sampling parameter
    - stop_sequences (List[str]): Sequences that stop generation
    - thinking_budget (int): Token budget for thinking models
    - include_thoughts (bool): Whether to include thinking in response
    """

    # Default endpoints
    ANTIGRAVITY_DAILY_API_BASE = "https://daily-cloudcode-pa.sandbox.googleapis.com"
    ANTIGRAVITY_PROD_API_BASE = "https://cloudcode-pa.googleapis.com"
    DEFAULT_PROJECT_ID = "rising-fact-p41fc"

    # OAuth configuration
    OAUTH_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    # Supported models
    SUPPORTED_MODELS = [
        # Claude models
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-thinking-low",
        "claude-sonnet-4-5-thinking-medium",
        "claude-sonnet-4-5-thinking-high",
        "claude-opus-4-5-thinking-low",
        "claude-opus-4-5-thinking-medium",
        "claude-opus-4-5-thinking-high",
        # Gemini models
        "gemini-3-pro-low",
        "gemini-3-pro-high",
        "gemini-3-flash",
        # GPT-OSS models
        "gpt-oss-120b-medium",
    ]

    # Thinking budget tiers
    THINKING_BUDGETS = {
        "low": 8192,
        "medium": 16384,
        "high": 32768,
    }

    # Configuration parameters
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    thinking_budget: Optional[int] = None
    include_thoughts: Optional[bool] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        thinking_budget: Optional[int] = None,
        include_thoughts: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    @staticmethod
    def ui_friendly_name() -> str:
        return "Antigravity (Google Unified Gateway)"

    def is_thinking_model(self, model: str) -> bool:
        """Check if the model is a thinking-enabled model."""
        return "thinking" in model.lower()

    def get_thinking_budget_from_model(self, model: str) -> Optional[int]:
        """Extract thinking budget from model name like 'claude-sonnet-4-5-thinking-high'."""
        for tier, budget in self.THINKING_BUDGETS.items():
            if f"thinking-{tier}" in model.lower():
                return budget
        return None

    def _is_claude_model(self, model: str) -> bool:
        """Check if the model is a Claude model."""
        return model.lower().startswith("claude")

    def _is_gemini_model(self, model: str) -> bool:
        """Check if the model is a Gemini model."""
        return model.lower().startswith("gemini")

    def get_supported_openai_params(self, model: str) -> List[str]:
        """Return list of supported OpenAI-compatible parameters."""
        supported_params = [
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stream",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "n",
        ]
        # Thinking parameters for thinking-enabled models
        if self.is_thinking_model(model):
            supported_params.extend(["thinking", "reasoning_effort"])
        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Map OpenAI parameters to Antigravity/Gemini format."""
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            elif param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_output_tokens"] = value
            elif param == "top_p":
                optional_params["top_p"] = value
            elif param == "stop":
                if isinstance(value, str):
                    optional_params["stop_sequences"] = [value]
                else:
                    optional_params["stop_sequences"] = value
            elif param == "n":
                optional_params["candidate_count"] = value
            elif param == "tools":
                optional_params["tools"] = value
            elif param == "tool_choice":
                optional_params["tool_choice"] = value
            elif param == "response_format":
                optional_params["response_format"] = value
            elif param == "stream":
                optional_params["stream"] = value
            elif param == "thinking":
                optional_params["thinking"] = value
            elif param == "reasoning_effort":
                optional_params["reasoning_effort"] = value

        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Validate environment and set up authentication headers.

        If no API key is provided, attempts to use the OAuth authenticator
        which will initiate device code flow if needed.
        """
        # Get API key (OAuth access token)
        api_key = api_key or get_secret_str("ANTIGRAVITY_API_KEY")

        if not api_key:
            # Try to get token via OAuth authenticator
            try:
                from litellm.llms.antigravity.authenticator import get_antigravity_api_key
                api_key = get_antigravity_api_key()
            except ImportError:
                pass
            except Exception as e:
                verbose_logger.warning(f"OAuth authentication failed: {e}")

        if not api_key:
            raise ValueError(
                "ANTIGRAVITY_API_KEY is not set. "
                "Please provide an OAuth access token via api_key parameter or "
                "ANTIGRAVITY_API_KEY environment variable, or run authentication flow. "
                "See: https://github.com/NoeFabris/opencode-antigravity-auth"
            )

        # Set authentication headers
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        headers["User-Agent"] = "litellm-antigravity/1.0"
        headers["X-Goog-Api-Client"] = "google-cloud-sdk litellm/0.1"
        headers["Client-Metadata"] = json.dumps({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI"
        })

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Construct the complete URL for Antigravity API."""
        if api_base is None:
            api_base = get_secret_str("ANTIGRAVITY_API_BASE") or self.ANTIGRAVITY_DAILY_API_BASE

        # Streaming vs non-streaming endpoint
        if stream:
            return f"{api_base}/v1internal:streamGenerateContent?alt=sse"
        else:
            return f"{api_base}/v1internal:generateContent"

    def _transform_messages_to_gemini_format(
        self, messages: List[AllMessageValues]
    ) -> List[dict]:
        """Transform OpenAI-style messages to Gemini-style contents array."""
        contents = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Map roles: OpenAI uses 'assistant', Gemini uses 'model'
            gemini_role = "model" if role == "assistant" else "user"

            # Handle system messages - skip them, they go into systemInstruction
            if role == "system":
                continue

            # Handle different content types
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Handle image content
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url
                            parts.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",  # Will be inferred by API
                                    "data": url if url.startswith("data:") else url
                                }
                            })
                    elif isinstance(item, str):
                        parts.append({"text": item})

            # Handle tool calls in assistant messages
            if role == "assistant" and "tool_calls" in message:
                tool_calls = message.get("tool_calls", [])
                for tool_call in tool_calls:
                    if tool_call.get("type") == "function":
                        func = tool_call.get("function", {})
                        parts.append({
                            "functionCall": {
                                "name": func.get("name", ""),
                                "args": json.loads(func.get("arguments", "{}")),
                                "id": tool_call.get("id", "")
                            }
                        })

            # Handle tool results
            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                name = message.get("name", "")
                parts = [{
                    "functionResponse": {
                        "name": name,
                        "id": tool_call_id,
                        "response": {"result": content}
                    }
                }]
                gemini_role = "user"

            if parts:
                contents.append({
                    "role": gemini_role,
                    "parts": parts
                })

        return contents

    def _get_system_instruction(
        self, messages: List[AllMessageValues]
    ) -> Optional[dict]:
        """Extract system instruction from messages."""
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, str):
                    return {
                        "parts": [{"text": content}]
                    }
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif isinstance(item, str):
                            parts.append({"text": item})
                    return {"parts": parts}
        return None

    def _transform_tools_to_gemini_format(
        self, tools: List[dict]
    ) -> Optional[List[dict]]:
        """Transform OpenAI-style tools to Gemini-style function declarations."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                # Validate function name (Antigravity rules)
                # Must start with letter or underscore, no slashes
                if name and not name[0].isalpha() and name[0] != "_":
                    verbose_logger.warning(
                        f"Function name '{name}' may not be valid for Antigravity API. "
                        "Names must start with a letter or underscore."
                    )
                # Replace slashes with underscores
                name = name.replace("/", "_")

                declaration = {
                    "name": name,
                    "description": func.get("description", ""),
                }
                if "parameters" in func:
                    parameters = self._clean_schema_for_antigravity(
                        func.get("parameters", {})
                    )
                    declaration["parameters"] = parameters
                function_declarations.append(declaration)

        if function_declarations:
            return [{"functionDeclarations": function_declarations}]
        return None

    def _clean_schema_for_antigravity(self, schema: dict) -> dict:
        """
        Clean JSON schema for Antigravity API compatibility.

        Removes unsupported fields:
        - $ref, $defs, definitions (inline instead)
        - $schema, $id
        - const (use enum with single value)
        - default, examples
        """
        if not isinstance(schema, dict):
            return schema

        cleaned = {}
        unsupported_keys = {
            "$ref", "$defs", "definitions", "$schema", "$id",
            "default", "examples"
        }

        for key, value in schema.items():
            if key in unsupported_keys:
                continue
            elif key == "const":
                # Convert const to enum with single value
                cleaned["enum"] = [value]
            elif key == "anyOf":
                cleaned["any_of"] = [
                    self._clean_schema_for_antigravity(v) for v in value
                ]
            elif key == "allOf":
                cleaned["all_of"] = [
                    self._clean_schema_for_antigravity(v) for v in value
                ]
            elif key == "oneOf":
                cleaned["one_of"] = [
                    self._clean_schema_for_antigravity(v) for v in value
                ]
            elif isinstance(value, dict):
                cleaned[key] = self._clean_schema_for_antigravity(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._clean_schema_for_antigravity(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                cleaned[key] = value

        return cleaned

    def _get_generation_config(
        self, model: str, optional_params: dict
    ) -> dict:
        """Build the generation config for the request."""
        config = {}

        if "temperature" in optional_params:
            config["temperature"] = optional_params["temperature"]
        if "max_output_tokens" in optional_params:
            config["maxOutputTokens"] = optional_params["max_output_tokens"]
        if "top_p" in optional_params:
            config["topP"] = optional_params["top_p"]
        if "top_k" in optional_params:
            config["topK"] = optional_params["top_k"]
        if "stop_sequences" in optional_params:
            config["stopSequences"] = optional_params["stop_sequences"]
        if "candidate_count" in optional_params:
            config["candidateCount"] = optional_params["candidate_count"]

        # Handle thinking configuration
        if self.is_thinking_model(model):
            thinking_budget = self.get_thinking_budget_from_model(model)
            if thinking_budget:
                config["thinkingConfig"] = {
                    "thinkingBudget": thinking_budget,
                    "includeThoughts": optional_params.get("include_thoughts", True)
                }
            # Override with explicit thinking params if provided
            if "thinking" in optional_params:
                thinking = optional_params["thinking"]
                if isinstance(thinking, dict):
                    if "budget_tokens" in thinking:
                        config.setdefault("thinkingConfig", {})
                        config["thinkingConfig"]["thinkingBudget"] = thinking["budget_tokens"]
                    if "type" in thinking:
                        config.setdefault("thinkingConfig", {})
                        config["thinkingConfig"]["includeThoughts"] = thinking["type"] == "enabled"

        return config

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Transform OpenAI request to Antigravity/Gemini format."""
        # Get project ID
        project_id = (
            litellm_params.get("project")
            or get_secret_str("ANTIGRAVITY_PROJECT_ID")
            or self.DEFAULT_PROJECT_ID
        )

        # Transform messages to Gemini format
        contents = self._transform_messages_to_gemini_format(messages)

        # Build the inner request object
        inner_request: Dict[str, Any] = {
            "contents": contents
        }

        # Add system instruction if present
        system_instruction = self._get_system_instruction(messages)
        if system_instruction:
            inner_request["systemInstruction"] = system_instruction

        # Add generation config
        generation_config = self._get_generation_config(model, optional_params)
        if generation_config:
            inner_request["generationConfig"] = generation_config

        # Add tools if present
        if "tools" in optional_params:
            tools = self._transform_tools_to_gemini_format(optional_params["tools"])
            if tools:
                inner_request["tools"] = tools

        # Build the full Antigravity request
        request_data = {
            "project": project_id,
            "model": model,
            "request": inner_request,
            "userAgent": "litellm-antigravity",
            "requestId": f"litellm-{str(uuid.uuid4())}"
        }

        return request_data

    def _parse_gemini_response_content(
        self, parts: List[dict]
    ) -> Tuple[str, List[ChatCompletionToolCallChunk], List[ChatCompletionThinkingBlock]]:
        """Parse Gemini response parts into content, tool calls, and thinking blocks."""
        text_content = ""
        tool_calls = []
        thinking_blocks = []

        for i, part in enumerate(parts):
            if "text" in part:
                # Check if this is a thinking block
                if part.get("thought", False) or part.get("thoughtSignature"):
                    thinking_blocks.append(
                        ChatCompletionThinkingBlock(
                            type="thinking",
                            thinking=part.get("text", ""),
                            signature=part.get("thoughtSignature"),
                        )
                    )
                else:
                    text_content += part.get("text", "")
            elif "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append(
                    ChatCompletionToolCallChunk(
                        id=func_call.get("id", f"call_{i}"),
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=func_call.get("name", ""),
                            arguments=json.dumps(func_call.get("args", {}))
                        ),
                        index=i,
                    )
                )

        return text_content, tool_calls, thinking_blocks

    def _map_finish_reason(self, gemini_reason: str) -> str:
        """Map Gemini finish reason to OpenAI format."""
        mapping = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop",
            "BLOCKLIST": "content_filter",
            "PROHIBITED_CONTENT": "content_filter",
            "SPII": "content_filter",
        }
        return mapping.get(gemini_reason, "stop")

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """Transform Antigravity/Gemini response to OpenAI format."""
        try:
            response_json = raw_response.json()
        except json.JSONDecodeError:
            raise AntigravityError(
                status_code=500,
                message=f"Failed to parse response JSON: {raw_response.text}",
                response=raw_response,
            )

        # Handle error responses
        if "error" in response_json:
            error = response_json["error"]
            raise AntigravityError(
                status_code=error.get("code", 500),
                message=error.get("message", "Unknown error"),
                response=raw_response,
                body=error,
            )

        # Extract the inner response
        inner_response = response_json.get("response", response_json)

        # Get candidates
        candidates = inner_response.get("candidates", [])
        if not candidates:
            raise AntigravityError(
                status_code=500,
                message="No candidates in response",
                response=raw_response,
            )

        # Process each candidate
        choices = []
        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            text_content, tool_calls, thinking_blocks = self._parse_gemini_response_content(parts)

            # Build the message
            message_dict: Dict[str, Any] = {
                "role": "assistant",
                "content": text_content if text_content else None,
            }

            if tool_calls:
                message_dict["tool_calls"] = [tc.model_dump() for tc in tool_calls]

            # Add thinking blocks if present (for Claude thinking models)
            if thinking_blocks and self._is_claude_model(model):
                message_dict["thinking_blocks"] = [tb.model_dump() for tb in thinking_blocks]

            finish_reason = self._map_finish_reason(
                candidate.get("finishReason", "STOP")
            )

            choice = Choices(
                index=i,
                message=Message(**message_dict),
                finish_reason=finish_reason,
            )
            choices.append(choice)

        # Extract usage metadata
        usage_metadata = inner_response.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0),
            completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0),
        )

        # Update model response
        model_response.choices = choices
        model_response.usage = usage
        model_response.model = inner_response.get("modelVersion", model)
        model_response.id = inner_response.get("responseId", f"chatcmpl-{str(uuid.uuid4())}")
        model_response.created = int(time.time())

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """Return appropriate error class for the given error."""
        return AntigravityError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """Get an iterator for streaming responses."""
        if sync_stream:
            return AntigravityStreamingIterator(
                streaming_response=streaming_response,
                json_mode=json_mode,
            )
        else:
            return AntigravityAsyncStreamingIterator(
                streaming_response=streaming_response,
                json_mode=json_mode,
            )


class AntigravityStreamingIterator:
    """Iterator for synchronous SSE streaming responses from Antigravity API."""

    def __init__(
        self,
        streaming_response: Iterator[str],
        json_mode: Optional[bool] = False,
    ):
        self.streaming_response = streaming_response
        self.json_mode = json_mode
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration

        try:
            chunk = next(self.streaming_response)
            return self._process_chunk(chunk)
        except StopIteration:
            self.finished = True
            raise

    def _process_chunk(self, chunk: str) -> dict:
        """Process a single SSE chunk from the streaming response."""
        # SSE format: data: {json}
        if chunk.startswith("data: "):
            chunk = chunk[6:]

        if not chunk.strip():
            return {"choices": [{"delta": {}}]}

        try:
            data = json.loads(chunk)
            response = data.get("response", data)
            candidates = response.get("candidates", [])

            if not candidates:
                return {"choices": [{"delta": {}}]}

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            delta: Dict[str, Any] = {}
            for part in parts:
                if "text" in part and not part.get("thought", False):
                    delta["content"] = part.get("text", "")
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    delta["tool_calls"] = [{
                        "index": 0,
                        "id": func_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": func_call.get("name", ""),
                            "arguments": json.dumps(func_call.get("args", {}))
                        }
                    }]

            finish_reason = None
            if "finishReason" in candidate:
                finish_reason = AntigravityConfig()._map_finish_reason(
                    candidate["finishReason"]
                )

            return {
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }]
            }

        except json.JSONDecodeError:
            return {"choices": [{"delta": {}}]}


class AntigravityAsyncStreamingIterator:
    """Async iterator for SSE streaming responses from Antigravity API."""

    def __init__(
        self,
        streaming_response: AsyncIterator[str],
        json_mode: Optional[bool] = False,
    ):
        self.streaming_response = streaming_response
        self.json_mode = json_mode
        self.finished = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.finished:
            raise StopAsyncIteration

        try:
            chunk = await self.streaming_response.__anext__()
            return self._process_chunk(chunk)
        except StopAsyncIteration:
            self.finished = True
            raise

    def _process_chunk(self, chunk: str) -> dict:
        """Process a single SSE chunk from the streaming response."""
        # SSE format: data: {json}
        if chunk.startswith("data: "):
            chunk = chunk[6:]

        if not chunk.strip():
            return {"choices": [{"delta": {}}]}

        try:
            data = json.loads(chunk)
            response = data.get("response", data)
            candidates = response.get("candidates", [])

            if not candidates:
                return {"choices": [{"delta": {}}]}

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            delta: Dict[str, Any] = {}
            for part in parts:
                if "text" in part and not part.get("thought", False):
                    delta["content"] = part.get("text", "")
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    delta["tool_calls"] = [{
                        "index": 0,
                        "id": func_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": func_call.get("name", ""),
                            "arguments": json.dumps(func_call.get("args", {}))
                        }
                    }]

            finish_reason = None
            if "finishReason" in candidate:
                finish_reason = AntigravityConfig()._map_finish_reason(
                    candidate["finishReason"]
                )

            return {
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }]
            }

        except json.JSONDecodeError:
            return {"choices": [{"delta": {}}]}
