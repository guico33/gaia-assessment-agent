"""
Centralized LLM provider instantiation for GAIA Agent.
"""

from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import Config


class LLMProviderError(Exception):
    """Raised when LLM provider cannot be instantiated."""

    pass


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> BaseChatModel:
    """
    Get a configured LLM instance.

    Args:
        provider: LLM provider ("openai" or "anthropic"). If None, uses auto-detection.
        model_name: Specific model name. If None, uses provider default.
        temperature: Model temperature. If None, uses config default.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Configured LLM instance

    Raises:
        LLMProviderError: If no valid provider can be instantiated
    """
    # Use provided values or defaults from config
    temperature = temperature if temperature is not None else Config.DEFAULT_TEMPERATURE

    # Determine provider
    if provider is None:
        provider = Config.get_default_provider()
        if provider is None:
            raise LLMProviderError(
                "No LLM provider available. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
            )

    # Validate provider is available
    available_providers = Config.get_available_providers()
    if provider not in available_providers:
        raise LLMProviderError(
            f"Provider '{provider}' not available. Available providers: {available_providers}. "
            f"Please check your API key environment variables."
        )

    # Instantiate the appropriate provider
    if provider == "openai":
        model = model_name or Config.DEFAULT_OPENAI_MODEL
        api_key = Config.get_openai_api_key()
        if not api_key:
            raise LLMProviderError("OPENAI_API_KEY environment variable not set")

        return ChatOpenAI(
            model=model, temperature=temperature, api_key=SecretStr(api_key), **kwargs
        )

    elif provider == "anthropic":
        model = model_name or Config.DEFAULT_ANTHROPIC_MODEL
        api_key = Config.get_anthropic_api_key()
        if not api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY environment variable not set")

        return ChatAnthropic(
            model_name=model,
            temperature=temperature,
            api_key=SecretStr(api_key),
            **kwargs,
        )

    else:
        raise LLMProviderError(f"Unsupported provider: {provider}")


def get_vision_llm(
    provider: Optional[str] = None, temperature: Optional[float] = None, **kwargs
) -> BaseChatModel:
    """
    Get a vision-capable LLM instance for image analysis.

    Args:
        provider: LLM provider. If None, uses auto-detection.
        temperature: Model temperature. If None, uses config default.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Vision-capable LLM instance

    Raises:
        LLMProviderError: If no valid vision provider can be instantiated
    """
    # For now, we primarily use OpenAI for vision tasks
    # You can extend this to support other vision-capable models

    if provider is None:
        # Prefer OpenAI for vision tasks, fallback to configured default
        if "openai" in Config.get_available_providers():
            provider = "openai"
        else:
            provider = Config.get_default_provider()

    if provider == "openai":
        # Use vision-capable OpenAI model
        return get_llm(
            provider="openai",
            model_name=Config.DEFAULT_OPENAI_MODEL,  # gpt-4o-mini has vision
            temperature=temperature,
            **kwargs,
        )
    elif provider == "anthropic":
        # Claude models also support vision
        return get_llm(
            provider="anthropic",
            model_name=Config.DEFAULT_ANTHROPIC_MODEL,  # Claude 3.5 Sonnet has vision
            temperature=temperature,
            **kwargs,
        )
    else:
        raise LLMProviderError(f"Vision not supported for provider: {provider}")


def get_analysis_llm(
    provider: Optional[str] = None, temperature: Optional[float] = None, **kwargs
) -> BaseChatModel:
    """
    Get an LLM instance optimized for data analysis tasks.

    Args:
        provider: LLM provider. If None, uses auto-detection.
        temperature: Model temperature. If None, uses config default.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Analysis-optimized LLM instance

    Raises:
        LLMProviderError: If no valid provider can be instantiated
    """
    # Use the same logic as general LLM but potentially with different temperature
    analysis_temperature = (
        temperature if temperature is not None else 0.1
    )  # Lower for analysis

    return get_llm(provider=provider, temperature=analysis_temperature, **kwargs)


def validate_llm_setup() -> None:
    """
    Validate that at least one LLM provider is properly configured.

    Raises:
        LLMProviderError: If no valid provider is available
    """
    try:
        # Try to get a default LLM instance
        get_llm()
    except LLMProviderError as e:
        raise LLMProviderError(
            f"LLM setup validation failed: {e}\n"
            f"Please ensure you have set either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        )


# Validate setup on module import
try:
    validate_llm_setup()
except LLMProviderError as e:
    print(f"⚠️  LLM Provider Warning: {e}")
    print("The application may not function correctly without proper API keys.")
