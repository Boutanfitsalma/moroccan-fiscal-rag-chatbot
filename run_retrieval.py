# run_retrieval.py
import sys
import json
from loguru import logger
from src.config import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from src.retriever import FiscalRetriever

# --- Configure Logger ---
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

def main():
    """
    Main function to initialize the retriever and test a query.
    """
    logger.info("--- Initializing the Fiscal Retriever ---")
    try:
        retriever = FiscalRetriever(
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Could not initialize retriever. Error: {e}")
        return

    # --- Example Queries ---
    #query = "Quelles sont les conditions pour les promoteurs immobiliers en 2011?"
    query = "quel est le taux d'impôt pour un résultat fiscal de 2000?"
    #query = "What is the tax rate for a fiscal result of 2000?" # Test multilingual
    
    # --- Run Retrieval ---
    # Example 1: Standard retrieval
    logger.info("\n--- Running Standard Retrieval ---")
    retrieved_docs = retriever.retrieve(query)

    # Example 2: Retrieval with a year filter
    # logger.info("\n--- Running Retrieval with Year Filter (2011) ---")
    # retrieved_docs = retriever.retrieve(query, year_filter="2011")

    # --- Display Results ---
    if not retrieved_docs:
        logger.warning("No documents were retrieved for the query.")
        return

    print("\n" + "="*20 + " RETRIEVED RESULTS " + "="*20)
    for i, doc in enumerate(retrieved_docs):
        # Create a clean display version
        display_doc = doc.copy()
        content = display_doc.pop('content', '')
        # Truncate content for display
        if len(content) > 300:
            content = content[:300] + "..."
        
        print(f"\n--- Document #{i+1} (ID: {display_doc.get('chunk_id')}) ---")
        # Pretty print metadata
        print(json.dumps(display_doc, indent=2, ensure_ascii=False))
        print(f"\nContent Snippet:\n{content}")
    print("\n" + "="*58)


if __name__ == "__main__":
    main()