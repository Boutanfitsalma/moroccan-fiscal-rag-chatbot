# run_chatbot.py
from loguru import logger
from src.pipeline import run_rag_pipeline
from src.config import MODEL_PAIRS, LOCAL_MODELS

def test_pipeline():
    """Function to test the RAG pipeline with different configurations."""
    
    # --- CHOOSE YOUR TEST CONFIGURATION HERE ---

    # === Test 1: OpenRouter (API-based) with a French query ===
    # provider = "openrouter"
    # model_id = MODEL_PAIRS["Duel Mistral"]["classic"]["id"]
    # query = "Quelles sont les conditions d'exonération pour les promoteurs immobiliers ?"
    
    # === Test 2: Ollama (Local) with a French query ===
    # provider = "ollama"
    # model_id = LOCAL_MODELS["mistral"]["id"]
    # query = "Quel est le traitement fiscal des produits de location d'aéronefs?"

    # === Test 3: Test with the French query about aircraft rental taxation ===
    provider = "openrouter"
    model_id = MODEL_PAIRS["Duel Qwen"]["classic"]["id"]
    query = "Quel est le traitement fiscal des produits de location d'aéronefs?"

    # --- EXECUTE THE PIPELINE ---
    logger.info(f"Running test with provider='{provider}', model='{model_id}'")
    final_response = run_rag_pipeline(
        query=query,
        provider=provider,
        model_id=model_id
    )

    # --- DISPLAY THE RESULT ---
    print("\n" + "="*30 + " FINAL RESPONSE " + "="*30)
    print(f"Query: {query}")
    print(f"Model: {provider} / {model_id}")
    print("-" * 80)
    print(final_response)
    print("="*80)

if __name__ == "__main__":
    test_pipeline()