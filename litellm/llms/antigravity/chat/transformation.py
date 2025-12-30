"""
Chat completion configuration and transformation for Antigravity API.

This module transforms OpenAI-style chat completion requests to Antigravity
format and handles response transformation back to OpenAI format.
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from litellm._logging import verbose_logger
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, Usage

from ..authenticator import Authenticator
from ..common_utils import (
    ANTIGRAVITY_API_BASE,
    ANTIGRAVITY_GENERATE_PATH,
    ANTIGRAVITY_STREAM_PATH,
    ANTHROPIC_BETA_HEADER,
    CLAUDE_THINKING_MAX_OUTPUT_TOKENS,
    AntigravityError,
    clean_json_schema,
    get_antigravity_default_headers,
    get_streaming_headers,
    get_thinking_budget,
    is_thinking_model,
    normalize_model_name,
)


# Finish reason mapping from Antigravity to OpenAI
FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "OTHER": "stop",  # Function calls often use OTHER
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
}


class AntigravityConfig(BaseConfig):
    """
    Configuration class for Antigravity API.

    Handles authentication, request transformation, and response parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "antigravity",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()
        self._custom_llm_provider = custom_llm_provider

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Get provider info for OpenAI-compatible routing.

        Returns:
            Tuple of (api_base, api_key, custom_llm_provider)

        Note: This method does NOT trigger OAuth flow. Authentication is deferred
        to validate_environment() which is called when actual requests are made.
        This allows the proxy to start without blocking on authentication.
        """
        dynamic_api_base = api_base or self.authenticator.get_api_base() or ANTIGRAVITY_API_BASE

        # Return placeholder - actual auth happens in validate_environment()
        # This allows proxy startup without blocking on OAuth
        dynamic_api_key = "deferred-auth"

        return dynamic_api_base, dynamic_api_key, custom_llm_provider

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
        """
        Validate environment and set up request headers.

        This triggers OAuth authentication if needed.
        """
        try:
            access_token = self.authenticator.get_access_token()
        except AntigravityError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=self._custom_llm_provider,
                message=str(e),
            )

        # Get default headers
        validated_headers = get_antigravity_default_headers(access_token)

        # Add Claude-specific headers for thinking models
        normalized_model = normalize_model_name(model)
        if self._is_claude_model(normalized_model) and is_thinking_model(normalized_model):
            validated_headers["anthropic-beta"] = ANTHROPIC_BETA_HEADER

        # Merge with any provided headers
        validated_headers.update(headers)

        return validated_headers

    def _is_claude_model(self, model: str) -> bool:
        """Check if the model is a Claude model."""
        return "claude" in model.lower()

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the API request.

        Args:
            api_base: Base URL for the API.
            api_key: API key (unused for Antigravity, uses OAuth).
            model: Model name.
            optional_params: Optional parameters.
            litellm_params: LiteLLM parameters.
            stream: Whether streaming is enabled.

        Returns:
            Complete URL for the request.
        """
        base = api_base or ANTIGRAVITY_API_BASE

        if stream:
            return f"{base}{ANTIGRAVITY_STREAM_PATH}?alt=sse"
        return f"{base}{ANTIGRAVITY_GENERATE_PATH}"

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for Antigravity.

        Args:
            model: Model name.

        Returns:
            List of supported parameter names.
        """
        base_params = [
            "messages",
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "stream",
            "tools",
            "tool_choice",
        ]

        # Add thinking parameters for thinking models
        if is_thinking_model(model):
            base_params.extend(["thinking", "reasoning_effort"])

        return base_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Antigravity parameters.

        Args:
            non_default_params: Non-default parameters from the request.
            optional_params: Optional parameters to populate.
            model: Model name.
            drop_params: Whether to drop unsupported parameters.

        Returns:
            Updated optional_params dict.
        """
        supported_params = self.get_supported_openai_params(model)

        for key, value in non_default_params.items():
            if key in supported_params:
                optional_params[key] = value
            elif not drop_params:
                optional_params[key] = value

        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> Dict[str, Any]:
        """
        Transform OpenAI-style request to Antigravity format.

        Args:
            model: Model name.
            messages: List of messages in OpenAI format.
            optional_params: Optional parameters.
            litellm_params: LiteLLM parameters.
            headers: Request headers.

        Returns:
            Request body in Antigravity format.
        """
        # Normalize model name
        normalized_model = normalize_model_name(model)

        # Get project ID
        project_id = self.authenticator.get_project_id()

        # Transform messages
        contents, system_instruction = self._transform_messages(messages)

        # Build generation config
        generation_config = self._build_generation_config(normalized_model, optional_params)

        # Build request
        request_body: Dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            request_body["systemInstruction"] = system_instruction

        if generation_config:
            request_body["generationConfig"] = generation_config

        # Transform tools if present
        tools = optional_params.get("tools")
        if tools:
            request_body["tools"] = self._transform_tools(tools)

            # Add tool config for Claude models (VALIDATED mode)
            tool_config = self._build_tool_config(normalized_model, tools)
            if tool_config:
                request_body["toolConfig"] = tool_config

        # Build full API body
        api_body = {
            "project": project_id,
            "model": normalized_model,
            "request": request_body,
            "userAgent": "antigravity",
            "requestId": str(uuid4()),
        }

        return api_body

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Antigravity format.

        Args:
            messages: Messages in OpenAI format.

        Returns:
            Tuple of (contents, system_instruction)
        """
        contents: List[Dict[str, Any]] = []
        system_instruction: Optional[Dict[str, Any]] = None

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # System messages become systemInstruction
                if isinstance(content, str):
                    system_instruction = {"parts": [{"text": content}]}
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            parts.append({"text": part["text"]})
                        elif isinstance(part, str):
                            parts.append({"text": part})
                    system_instruction = {"parts": parts}
                continue

            # Map roles
            antigravity_role = "model" if role == "assistant" else "user"

            # Transform content to parts
            parts = self._transform_content_to_parts(content, message)

            if parts:
                contents.append({
                    "role": antigravity_role,
                    "parts": parts,
                })

        return contents, system_instruction

    def _transform_content_to_parts(
        self, content: Union[str, List[Any]], message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Transform message content to Antigravity parts format.

        Args:
            content: Message content (string or list of content blocks).
            message: Full message dict for additional context.

        Returns:
            List of parts in Antigravity format.
        """
        parts: List[Dict[str, Any]] = []

        # Handle tool calls from assistant
        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                parts.append({
                    "functionCall": {
                        "name": function.get("name", ""),
                        "args": json.loads(function.get("arguments", "{}")),
                        "id": tool_call.get("id", str(uuid4())),
                    }
                })

        # Handle tool response from user
        if message.get("role") == "tool":
            parts.append({
                "functionResponse": {
                    "name": message.get("name", ""),
                    "id": message.get("tool_call_id", ""),
                    "response": json.loads(content) if isinstance(content, str) else content,
                }
            })
            return parts

        # Handle string content
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
            return parts

        # Handle list content (multi-modal)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict):
                    if "text" in item:
                        parts.append({"text": item["text"]})
                    elif item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Handle image URLs (base64 or URL)
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                        if url.startswith("data:"):
                            # Base64 encoded image
                            parts.append({"inlineData": {"data": url.split(",")[1], "mimeType": url.split(";")[0].split(":")[1]}})
                        else:
                            parts.append({"fileData": {"fileUri": url}})

        return parts

    def _build_generation_config(
        self, model: str, optional_params: dict
    ) -> Dict[str, Any]:
        """
        Build generation config from optional parameters.

        Args:
            model: Normalized model name.
            optional_params: Optional parameters.

        Returns:
            Generation config dict.
        """
        config: Dict[str, Any] = {}

        if "max_tokens" in optional_params:
            config["maxOutputTokens"] = optional_params["max_tokens"]

        if "temperature" in optional_params:
            config["temperature"] = optional_params["temperature"]

        if "top_p" in optional_params:
            config["topP"] = optional_params["top_p"]

        if "stop" in optional_params:
            stops = optional_params["stop"]
            if isinstance(stops, str):
                config["stopSequences"] = [stops]
            else:
                config["stopSequences"] = stops

        # Add thinking config for thinking models
        if is_thinking_model(model):
            thinking_budget = get_thinking_budget(model)
            if thinking_budget:
                config["thinkingConfig"] = {
                    "thinkingBudget": thinking_budget,
                    "includeThoughts": True,
                }
                # Ensure maxOutputTokens is greater than thinkingBudget (per reference)
                if "maxOutputTokens" not in config or config["maxOutputTokens"] < thinking_budget:
                    config["maxOutputTokens"] = max(
                        config.get("maxOutputTokens", 0),
                        CLAUDE_THINKING_MAX_OUTPUT_TOKENS,
                    )

        return config

    def _build_tool_config(self, model: str, tools: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        Build tool config for Claude models.

        Claude models require toolConfig.functionCallingConfig.mode = "VALIDATED"
        for strict parameter validation.
        """
        if not tools:
            return None

        if self._is_claude_model(model):
            return {
                "functionCallingConfig": {
                    "mode": "VALIDATED"
                }
            }
        return None

    def _transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform OpenAI tools to Antigravity format.

        Args:
            tools: Tools in OpenAI format.

        Returns:
            Tools in Antigravity format.
        """
        function_declarations = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})

                # Clean the parameters schema
                parameters = function.get("parameters", {})
                cleaned_parameters = clean_json_schema(parameters)

                function_declarations.append({
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": cleaned_parameters,
                })

        return [{"functionDeclarations": function_declarations}]

    def transform_response(
        self,
        model: str,
        raw_response: Any,
        model_response: Any,
        logging_obj: Any,
        request_data: Dict[str, Any],
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> Any:
        """
        Transform Antigravity response to OpenAI format.

        Args:
            model: Model name.
            raw_response: Raw API response (httpx.Response object).
            model_response: Model response object to populate.
            logging_obj: Logging object.
            request_data: Original request data.
            messages: Original messages.
            optional_params: Optional parameters.
            litellm_params: LiteLLM parameters.
            encoding: Token encoding.
            api_key: API key.
            json_mode: Whether JSON mode is enabled.

        Returns:
            Populated model response.
        """
        # Parse JSON from httpx.Response
        try:
            response_json = raw_response.json()
        except Exception:
            response_json = {}

        inner_response = response_json.get("response", {})
        candidates = inner_response.get("candidates", [])

        if not candidates:
            verbose_logger.warning("No candidates in Antigravity response")
            return model_response

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text and function calls
        text_content = ""
        tool_calls = []

        for part in parts:
            if "text" in part and not part.get("thought"):
                text_content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": fc.get("id", str(uuid4())),
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    }
                })

        # Build message using LiteLLM Message type
        message = Message(
            role="assistant",
            content=text_content if text_content else None,
        )

        if tool_calls:
            message.tool_calls = tool_calls

        # Map finish reason
        finish_reason = FINISH_REASON_MAP.get(
            candidate.get("finishReason", "STOP"), "stop"
        )

        # Build choice using LiteLLM Choices type
        choice = Choices(
            finish_reason=finish_reason,
            index=0,
            message=message,
        )

        model_response.choices = [choice]

        # Extract usage using LiteLLM Usage type
        usage_metadata = inner_response.get("usageMetadata", {})
        model_response.usage = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0),
            completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0),
        )

        # Add thinking tokens if present
        if "thoughtsTokenCount" in usage_metadata:
            model_response.usage.reasoning_tokens = usage_metadata["thoughtsTokenCount"]

        # Set model info
        model_response.model = inner_response.get("modelVersion", model)

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Dict[str, Any]
    ) -> Exception:
        """
        Get appropriate exception class for error response.

        Args:
            error_message: Error message.
            status_code: HTTP status code.
            headers: Response headers.

        Returns:
            Exception instance.
        """
        from litellm.exceptions import (
            APIError,
            AuthenticationError,
            RateLimitError,
            ServiceUnavailableError,
        )

        if status_code == 401:
            return AuthenticationError(
                message=error_message,
                model="",
                llm_provider="antigravity",
            )
        elif status_code == 429:
            return RateLimitError(
                message=error_message,
                model="",
                llm_provider="antigravity",
            )
        elif status_code >= 500:
            return ServiceUnavailableError(
                message=error_message,
                model="",
                llm_provider="antigravity",
            )
        else:
            return APIError(
                message=error_message,
                status_code=status_code,
                model="",
                llm_provider="antigravity",
            )
