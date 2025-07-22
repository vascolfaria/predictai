"""
Model utility functions for the AI Hackathon Starter.

This module provides easy-to-use functions for initializing and working with
different LLM providers (OpenAI, Google Gemini) through LangChain.
"""

from typing import Optional, Any

# Import dotenv for loading environment variables
from dotenv import load_dotenv

# Import LangChain's chat models
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()


def init_model(
        model_name: Optional[str] = None,
        provider: str = "openai",
        temperature: float = 0.7,
        **kwargs
) -> Any:
    """
    Initialize a chat model using the specified provider.

    You can find the models you can select here:
        OpenAI: https://platform.openai.com/docs/models

    Args:
        model_name: The name of the model to use. If None, a default model for the provider will be used.
        provider: The provider to use ("openai").
        temperature: The temperature to use for generation (0.0 to 1.0).
        **kwargs: Additional arguments to pass to the model initialization.

    Returns:
        A LangChain chat model instance.

    Examples:
        >>> model = init_model(provider="openai")
        >>> model = init_model(model_name="gpt-4o", provider="openai", temperature=0.5)
    """
    # Define default models for each provider
    default_models = {
        "openai": "gpt-4.1",
    }

    # Use default model if none specified
    if model_name is None:
        model_name = default_models.get(provider, default_models["openai"])

    # Initialize the appropriate model based on provider
    if provider == "openai":
        # Extract just the model name if it has a provider prefix
        if ":" in model_name:
            model_name = model_name.split(":")[1]
        return ChatOpenAI(model_name=model_name, temperature=temperature, **kwargs)

    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai'")


def init_chat_model(
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
) -> Any:
    """
    Alias for init_model that's more descriptive for chat models.
    """
    return init_model(model_name=model_name, provider=provider, temperature=temperature, **kwargs)


def init_vision_model(
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
) -> Any:
    """
    Initialize a vision-capable model for image analysis.

    Args:
        provider: The provider to use ("openai" or "google_genai").
        model_name: The name of the model to use. If None, a default vision model for the provider will be used.
        temperature: The temperature to use for generation (0.0 to 1.0).
        **kwargs: Additional arguments to pass to the model initialization.

    Returns:
        A LangChain chat model instance with vision capabilities.

    Examples:
        >>> model = init_vision_model()  # Uses OpenAI's GPT-4o by default
    """
    # Define default vision models for each provider
    vision_models = {
        "openai": "gpt-4o"
    }

    # Use default vision model if none specified
    if model_name is None:
        model_name = vision_models.get(provider, vision_models["openai"])

    # Initialize and return the vision model with the same parameters
    return init_model(model_name=model_name, provider=provider, temperature=temperature, **kwargs)