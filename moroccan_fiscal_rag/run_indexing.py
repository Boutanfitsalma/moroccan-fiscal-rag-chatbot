# run_indexing.py
import sys
import json
from loguru import logger
from src.config import (
    DATA_DIR,
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    INDEXING_BATCH_SIZE
)
from src.data_loader import load_and_prepare_chunks
from src.indexer import VectorIndexer

# --- Configure Logger ---
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

def display_sample_chunks(chunks, count=3):
    """Prints a few sample chunks to verify the data structure."""
    logger.info(f"--- Displaying {count} Sample Prepared Chunks ---")
    for i, chunk in enumerate(chunks[:count]):
        # Create a copy to avoid modifying the original data
        display_chunk = chunk.copy()
        
        # Truncate long content for readability
        content = display_chunk.get('content', '')
        if len(content) > 150:
            display_chunk['content'] = content[:150] + "..."
            
        print(f"\n--- Sample Chunk #{i+1} ---")
        # Use json.dumps for pretty printing
        print(json.dumps(display_chunk, indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")


def main():
    """
    Main function to run the complete indexing pipeline.
    """
    logger.info("--- Starting Moroccan Fiscal Law Indexing Pipeline ---")

    # 1. Load, prepare, and de-duplicate chunks from the data directory
    chunks = load_and_prepare_chunks(DATA_DIR)

    if not chunks:
        logger.error("No chunks were loaded. Exiting pipeline.")
        return

    # 2. Display sample chunks for verification
    display_sample_chunks(chunks)

    # 3. Initialize the vector indexer
    indexer = VectorIndexer(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        model_name=EMBEDDING_MODEL_NAME,
        force_reindex=True
    )

    # 4. Run the indexing process
    indexer.index(chunks, batch_size=INDEXING_BATCH_SIZE)

    logger.info("--- Indexing Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()