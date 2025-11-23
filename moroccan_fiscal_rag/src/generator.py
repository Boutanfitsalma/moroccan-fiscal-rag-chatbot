# src/generator.py
import os
import re
import time
import requests
from typing import List, Dict, Any, Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from dotenv import load_dotenv

from src.config import SYSTEM_PROMPT_AR, SYSTEM_PROMPT_FR

def detect_language(text: str) -> str:
    """Detects if the text contains Arabic characters."""
    arabic_range = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    return 'ar' if arabic_range.search(text) else 'fr'

class FiscalResponseGenerator:
    """
    Generates final answers using different LLM providers (OpenRouter API or local Ollama).
    This class is now self-contained and does not require a pre-loaded LLM.
    """
    def __init__(self):
        logger.success("Generator initialized. Ready to handle OpenRouter and Ollama providers.")
        load_dotenv()
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Formats the list of context chunks into a structured string for the LLM."""
        if not context_chunks:
            return "Aucun contexte pertinent n'a été trouvé."

        context_str = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk.get('source', 'Source inconnue')
            year = chunk.get('year', 'N/A')
            chunk_id = chunk.get('chunk_id', 'N/A')
            
            context_str += f"--- Contexte #{i+1} (Source: {source} {year}, ID: {chunk_id}) ---\n"
            
            hierarchy_parts = [chunk.get(k) for k in ['roman_section_title', 'numeric_subsection_title', 'sub_topic'] if chunk.get(k)]
            if hierarchy_parts:
                context_str += f"Section: {' -> '.join(hierarchy_parts)}\n"

            content = chunk.get('content', 'Contenu non disponible.')
            context_str += f"Contenu:\n{content}\n\n"
            
        return context_str.strip()

    def _call_openrouter(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        """Handles the direct API call to OpenRouter."""
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file.")
        
        logger.info(f"Calling OpenRouter model: {model_id}")
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ], "temperature": 0.0, "max_tokens": 4096,
                }, timeout=180
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            error_details = e.response.json().get('error', {}).get('message', e.response.text)
            logger.error(f"HTTP Error calling OpenRouter ({e.response.status_code}): {error_details}")
            return f"❌ Erreur API: {error_details}"
        except Exception as e:
            logger.error(f"An unexpected exception occurred with OpenRouter: {e}")
            return f"❌ Exception: {e}"
            
    def _call_ollama(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        """Handles the local call to an Ollama model via LangChain."""
        logger.info(f"Calling local Ollama model: {model_id}")
        try:
            llm = ChatOllama(model=model_id, temperature=0.0)
            
            # Combine system and user prompts for the LangChain template
            full_prompt_template = system_prompt + "\n\n" + user_prompt
            
            prompt = ChatPromptTemplate.from_template(full_prompt_template)
            chain = prompt | llm | StrOutputParser()
            
            # Ollama doesn't need context/query separation in the invoke call
            # because we've built it all into the template.
            return chain.invoke({})
        except Exception as e:
            logger.error(f"An error occurred with Ollama: {e}")
            return (f"❌ Erreur lors de l'appel du modèle local Ollama '{model_id}'. "
                    "Assurez-vous que Ollama est en cours d'exécution et que le modèle est bien téléchargé (ex: `ollama run {model_id}`).")

    def generate(self, query: str, context: List[Dict[str, Any]], provider: str, model_id: str) -> str:
        """
        Generates the final response by routing to the correct LLM provider.
        """
        lang = detect_language(query)
        system_prompt = SYSTEM_PROMPT_AR if lang == 'ar' else SYSTEM_PROMPT_FR
        logger.info(f"Detected language: '{lang.upper()}'. Using corresponding system prompt.")

        formatted_context = self._format_context(context)
        user_prompt = f"CONTEXTE FOURNI:\n{formatted_context}\n\nQUESTION DE L'UTILISATEUR:\n{query}"

        if provider == 'openrouter':
            return self._call_openrouter(model_id, system_prompt, user_prompt)
        elif provider == 'ollama':
            return self._call_ollama(model_id, system_prompt, user_prompt)
        else:
            raise ValueError(f"Provider '{provider}' is not supported. Use 'openrouter' or 'ollama'.")