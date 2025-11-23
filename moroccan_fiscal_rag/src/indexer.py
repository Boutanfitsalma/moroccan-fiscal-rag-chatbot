# src/indexer.py
import json
from typing import List, Dict, Any

import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class VectorIndexer:
    """
    Handles the embedding and indexing of document chunks into a persistent
    ChromaDB vector store.
    """
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        model_name: str,
        force_reindex: bool = True
    ):
        """
        Initializes the indexer.

        Args:
            db_path (str): Path to the ChromaDB persistent storage directory.
            collection_name (str): Name of the collection to use.
            model_name (str): Name of the Sentence Transformer model.
            force_reindex (bool): If True, deletes any existing collection with the same name.
        """
        self.db_path = db_path
        self.collection_name = collection_name

        logger.info(f"Initializing embedding model: '{model_name}'...")
        # You can add {'device': 'cuda'} if a GPU is available
        self.embedding_model = SentenceTransformer(model_name)
        logger.success(f"Embedding model '{model_name}' loaded.")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)

        if force_reindex:
            self._recreate_collection()
        else:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Attached to existing collection '{self.collection_name}'.")

        logger.info(f"Indexer initialized. DB path: '{db_path}'")

    def _recreate_collection(self):
        """Deletes and recreates the collection for a fresh start."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Deleted existing collection: '{self.collection_name}'")
        except Exception:
            logger.info(f"Collection '{self.collection_name}' did not exist. Creating new one.")

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.success(f"Successfully created new collection: '{self.collection_name}'")

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serializes complex metadata values (lists, dicts) into JSON strings
        to ensure compatibility with ChromaDB.
        """
        serialized = {}
        for k, v in metadata.items():
            if isinstance(v, (list, dict)):
                serialized[k] = json.dumps(v, ensure_ascii=False)
            elif v is None:
                continue # Skip None values
            else:
                serialized[k] = v
        return serialized

    def index(self, chunks: List[Dict], batch_size: int = 32):
        """
        Embeds and indexes a list of prepared chunks into ChromaDB.

        Args:
            chunks (List[Dict]): A list of chunk dictionaries to index.
            batch_size (int): The number of chunks to process in each batch.
        """
        if not chunks:
            logger.warning("No chunks provided to index. Aborting.")
            return

        logger.info(f"Starting indexing of {len(chunks)} chunks into '{self.collection_name}'...")
        initial_count = self.collection.count()

        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing Batches"):
            batch = chunks[i:i + batch_size]

            # Prepare data for ChromaDB
            ids = [chunk["chunk_id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            
            # Metadata includes everything except the ID and content
            metadatas = [
                self._serialize_metadata({k: v for k, v in chunk.items() if k not in ['chunk_id', 'content']})
                for chunk in batch
            ]

            # Generate embeddings for the documents
            embeddings = self.embedding_model.encode(
                documents, show_progress_bar=False, normalize_embeddings=True
            ).tolist()

            # Add the batch to the collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

        final_count = self.collection.count()
        added_count = final_count - initial_count
        logger.success(f"Indexing complete. Added {added_count} new documents.")
        logger.info(f"Total documents in collection: {final_count}")