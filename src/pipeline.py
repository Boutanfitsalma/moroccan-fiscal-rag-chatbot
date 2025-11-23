# src/pipeline.py
from loguru import logger
from typing import List, Dict, Any

from src.config import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from src.retriever import FiscalRetriever
from src.generator import FiscalResponseGenerator

# --- Initialize Singleton instances for performance ---
# These objects are loaded once when the application starts.
try:
    logger.info("Initializing RAG pipeline components...")
    RETRIEVER = FiscalRetriever(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    GENERATOR = FiscalResponseGenerator()
    logger.success("RAG pipeline components initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize RAG components: {e}")
    RETRIEVER, GENERATOR = None, None

def run_rag_pipeline(query: str, provider: str, model_id: str) -> str:
    """
    Executes the full RAG pipeline: retrieve context and generate a response.
    This is the main entry point for any UI or API.

    Args:
        query (str): The user's question.
        provider (str): The LLM provider ('openrouter' or 'ollama').
        model_id (str): The specific model identifier.

    Returns:
        str: The generated answer from the LLM.
    """
    if not RETRIEVER or not GENERATOR:
        return "❌ Erreur: Le système RAG n'a pas été initialisé correctement. Veuillez vérifier les logs."

    logger.info(f"--- Starting RAG pipeline for query: '{query}' ---")
    
    # 1. Retrieve relevant context
    try:
        retrieved_context = RETRIEVER.retrieve(query, top_n_final=7)
        print(f"docs : {retrieved_context[1]}")
        if not retrieved_context:
            logger.warning("No relevant context was found for the query.")
            return "Je n'ai trouvé aucune information pertinente pour répondre à votre question dans les documents."
    except Exception as e:
        logger.error(f"An error occurred during retrieval: {e}")
        return "❌ Erreur: Une erreur s'est produite lors de la recherche d'informations."
    
    # 2. Generate a response
    try:
        final_response = GENERATOR.generate(
            query=query,
            context=retrieved_context,
            provider=provider,
            model_id=model_id 
        )
        return final_response
    except Exception as e:
        logger.error(f"An error occurred during generation: {e}")
        return "❌ Erreur: Une erreur s'est produite lors de la génération de la réponse."