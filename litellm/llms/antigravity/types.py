"""
Type definitions for Antigravity API.
"""
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class AntigravityContentPart(TypedDict, total=False):
    """A part of content in Antigravity format."""

    text: str
    functionCall: Dict[str, Any]
    functionResponse: Dict[str, Any]
    thought: bool
    thoughtSignature: str


class AntigravityContent(TypedDict):
    """Content in Antigravity format."""

    role: Literal["user", "model"]
    parts: List[AntigravityContentPart]


class AntigravitySystemInstruction(TypedDict):
    """System instruction in Antigravity format."""

    parts: List[Dict[str, str]]


class AntigravityThinkingConfig(TypedDict, total=False):
    """Thinking configuration for Claude/Gemini thinking models."""

    thinkingBudget: int
    includeThoughts: bool
    thinkingLevel: Literal["low", "high"]  # For Gemini
    maxThinkingLength: int  # For Gemini


class AntigravityGenerationConfig(TypedDict, total=False):
    """Generation configuration for Antigravity API."""

    maxOutputTokens: int
    temperature: float
    topP: float
    topK: int
    thinkingConfig: AntigravityThinkingConfig
    stopSequences: List[str]


class AntigravityFunctionDeclaration(TypedDict):
    """Function declaration for tools."""

    name: str
    description: str
    parameters: Dict[str, Any]


class AntigravityTool(TypedDict):
    """Tool definition for Antigravity API."""

    functionDeclarations: List[AntigravityFunctionDeclaration]


class AntigravityRequest(TypedDict, total=False):
    """Inner request payload for Antigravity API."""

    contents: List[AntigravityContent]
    systemInstruction: AntigravitySystemInstruction
    generationConfig: AntigravityGenerationConfig
    tools: List[AntigravityTool]


class AntigravityAPIBody(TypedDict):
    """Full API request body for Antigravity."""

    project: str
    model: str
    request: AntigravityRequest
    userAgent: str
    requestId: str


class AntigravityFunctionCall(TypedDict):
    """Function call from model."""

    name: str
    args: Dict[str, Any]
    id: str


class AntigravityResponsePart(TypedDict, total=False):
    """A part of response content."""

    text: str
    functionCall: AntigravityFunctionCall
    thought: bool
    thoughtSignature: str


class AntigravityResponseContent(TypedDict):
    """Response content from model."""

    role: Literal["model"]
    parts: List[AntigravityResponsePart]


class AntigravityCandidate(TypedDict, total=False):
    """A candidate response."""

    content: AntigravityResponseContent
    finishReason: str


class AntigravityUsageMetadata(TypedDict, total=False):
    """Usage metadata from response."""

    promptTokenCount: int
    candidatesTokenCount: int
    totalTokenCount: int
    thoughtsTokenCount: int


class AntigravityInnerResponse(TypedDict, total=False):
    """Inner response from Antigravity API."""

    candidates: List[AntigravityCandidate]
    usageMetadata: AntigravityUsageMetadata
    modelVersion: str
    responseId: str


class AntigravityAPIResponse(TypedDict, total=False):
    """Full API response from Antigravity."""

    response: AntigravityInnerResponse
    traceId: str


class AntigravityErrorDetail(TypedDict, total=False):
    """Error detail in error response."""

    type: str  # @type field
    retryDelay: str  # For rate limit responses


class AntigravityError(TypedDict, total=False):
    """Error response from Antigravity API."""

    code: int
    message: str
    status: str
    details: List[AntigravityErrorDetail]


class AntigravityErrorResponse(TypedDict):
    """Error response wrapper."""

    error: AntigravityError
