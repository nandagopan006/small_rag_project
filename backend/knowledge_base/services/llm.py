"""Service for generating college policy answers using OpenRouter LLM."""

import logging

import httpx
from django.conf import settings

from .prompts import POLICY_ASSISTANT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Raised when an error occurs during communication with OpenRouter."""

    pass


class LLMConfigurationError(LLMServiceError):
    """Raised when OpenRouter API configuration is missing or invalid."""

    pass


def get_llm_config() -> dict:
    """Read and validate OpenRouter settings from Django configuration.

    Returns:
        dict: Validated configuration with api_key, base_url, model, and timeout.

    Raises:
        LLMConfigurationError: If any required setting is missing or invalid.
    """
    api_key = getattr(settings, "OPENROUTER_API_KEY", "")
    if (
        not api_key
        or str(api_key).strip()
        in {"", "your_actual_openrouter_api_key", "your_openrouter_api_key"}
    ):
        raise LLMConfigurationError(
            "OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in your environment."
        )

    base_url = getattr(settings, "OPENROUTER_BASE_URL", "")
    if not base_url or not str(base_url).strip():
        raise LLMConfigurationError("OpenRouter base URL is not configured.")

    model_id = getattr(settings, "OPENROUTER_MODEL", "")
    if not model_id or not str(model_id).strip():
        raise LLMConfigurationError("OpenRouter model ID is not configured.")

    timeout_seconds = getattr(settings, "OPENROUTER_TIMEOUT", 30)
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise LLMConfigurationError(
            "OpenRouter request timeout must be a positive integer."
        )

    return {
        "api_key": str(api_key).strip(),
        "base_url": str(base_url).strip().rstrip("/"),
        "model": str(model_id).strip(),
        "timeout": timeout_seconds,
    }


def format_user_prompt(question: str = "", context: str = "", query: str = "") -> str:
    """Format user question and retrieved policy context into the user prompt."""
    prompt_question = (question or query).strip()
    return (
        f"Retrieved Policy Context:\n{context.strip()}\n\n"
        f"Question:\n{prompt_question}"
    )


def generate_answer(
    question: str = "",
    context: str = "",
    query: str = "",
    model: str | None = None,
    timeout: int | None = None,
    system_prompt: str = POLICY_ASSISTANT_SYSTEM_PROMPT,
) -> str:
    """Send user question and retrieved context to OpenRouter and return the answer.

    Args:
        question: The user's question about college policy.
        context: Formatted context string from retrieved chunks.
        query: Optional alias for question.
        model: Optional model identifier override.
        timeout: Optional request timeout in seconds.
        system_prompt: Optional custom system instructions.

    Returns:
        str: Generated answer from the model.

    Raises:
        ValueError: If question or context is empty.
        LLMConfigurationError: If OpenRouter settings are missing or invalid.
        LLMServiceError: If API call fails, times out, or returns invalid data.
    """

    #Validate the question
    raw_question = question or query
    if not raw_question or not isinstance(raw_question, str) or not raw_question.strip():
        raise ValueError("Question cannot be empty.")

    #Validate the retrieved context
    if not context or not isinstance(context, str) or not context.strip():
        raise ValueError("Context cannot be empty.")

    clean_question = raw_question.strip()
    clean_context = context.strip()


    config = get_llm_config()
    active_model = model.strip() if model and isinstance(model, str) and model.strip() else config["model"]
    active_timeout = timeout if (timeout is not None and isinstance(timeout, int) and timeout > 0) else config["timeout"]


    endpoint_url = f"{config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://collegepolicy.local",
        "X-Title": "College Policy RAG Assistant",
    }


    payload = {
        "model": active_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_user_prompt(clean_question, clean_context)},
        ],
        "temperature": 0.1,
    }


    try:
        with httpx.Client(timeout=active_timeout) as client:
            response = client.post(endpoint_url, headers=headers, json=payload)

    except httpx.TimeoutException as exc:
        logger.warning("OpenRouter request timed out after %s seconds.", active_timeout)
        raise LLMServiceError(
            "The request to the language model provider timed out. Please try again."
        ) from exc

    except (httpx.ConnectError, httpx.NetworkError) as exc:
        logger.warning("Network connection failure communicating with OpenRouter.")
        raise LLMServiceError(
            "Unable to connect to the language model provider. Please check network connectivity."
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error while sending request to OpenRouter.")
        raise LLMServiceError(
            "An unexpected error occurred while communicating with the language model provider."
        ) from exc


    if response.status_code != 200:
        logger.error("OpenRouter API returned error status: %s", response.status_code)
        raise LLMServiceError(
            f"Language model provider returned an error (status {response.status_code})."
        )


    try:
        response_data = response.json()
    except ValueError as exc:
        logger.error("Failed to parse JSON response from OpenRouter.")
        raise LLMServiceError(
            "Received malformed response from the language model provider."
        ) from exc

    #Extract and validate the generated answer
    choices = response_data.get("choices")
    if not choices or not isinstance(choices, list):
        raise LLMServiceError(
            "Language model provider returned an invalid response structure."
        )

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMServiceError(
            "Language model provider returned an invalid choice structure."
        )

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMServiceError(
            "Language model provider returned an invalid message structure."
        )

    answer_text = message.get("content")
    if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
        raise LLMServiceError(
            "Language model provider returned an empty response."
        )

    return answer_text.strip()


# Afor undeerstaning easy to in  backward compatibility
generate_policy_answer = generate_answer
