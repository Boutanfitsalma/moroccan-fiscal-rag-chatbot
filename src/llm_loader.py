# src/llm_loader.py
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from loguru import logger
import os
from dotenv import load_dotenv

def load_llm(provider: str = "openai", model_name: str = None) -> BaseChatModel:
    """
    Loads a LangChain-compatible chat model based on the specified provider.

    Args:
        provider (str): The LLM provider. Supported: "openai", "ollama".
        model_name (str): The specific model name to use.

    Returns:
        An instance of a BaseChatModel.
    """
    logger.info(f"Attempting to load LLM from provider: '{provider}'")
    
    load_dotenv()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        
        model = model_name or "gpt-3.5-turbo"
        logger.info(f"Loading OpenAI model: {model}")
        return ChatOpenAI(model_name=model, temperature=0.0, api_key=api_key)

    elif provider == "ollama":
        # Make sure Ollama service is running on your machine
        # Download a model first, e.g., `ollama run mistral`
        model = model_name or "mistral"
        logger.info(f"Loading Ollama model: {model}")
        return ChatOllama(model=model, temperature=0.0)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file.")
        logger.info("Provider 'openrouter' selected. Direct API calls will be used.")
        return None # No LangChain object needed
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'.")