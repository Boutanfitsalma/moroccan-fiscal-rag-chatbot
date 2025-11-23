# src/retriever.py
from typing import List, Dict, Optional, Any
import chromadb
from sentence_transformers import SentenceTransformer , CrossEncoder
from loguru import logger

# Placeholder for Query Expansion - can be replaced with a more sophisticated LLM-based one
class SimpleQueryExpander:
    """
    A basic query expander that generates a hypothetical answer.
    This can be swapped with a more complex model if needed.
    """
    def expand(self, query: str) -> List[str]:
        # A simple technique: combine original query with a hypothetical document.
        # This helps retrieve documents that answer the question.
        hypothetical_answer = f"La réponse à la question '{query}' est"
        return [query, f"{query} {hypothetical_answer}"]

class FiscalRetriever:
    """
    Handles retrieval from the fiscal vector database, with optional
    query expansion, metadata filtering, and re-ranking.
    """
    def __init__(self, db_path: str, collection_name: str, embedding_model_name: str):
        logger.info("Initializing FiscalRetriever...")
        
        # --- Connect to the existing Vector DB ---
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.success(f"Successfully connected to collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to connect to collection '{collection_name}'. Make sure indexing is complete.")
            raise e

        # --- Load Models ---
        logger.info(f"Loading embedding model: '{embedding_model_name}'")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        logger.info("Loading re-ranking model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # --- Initialize Components ---
        self.query_expander = SimpleQueryExpander()
        
        logger.success("FiscalRetriever initialized successfully.")

    def _rerank_documents(self, original_query: str, chunks: List[Dict]) -> List[Dict]:
        """Re-ranks a list of chunks based on their relevance to the original query."""
        if not chunks or not original_query:
            return []

        logger.debug(f"Re-ranking {len(chunks)} documents...")
        # The CrossEncoder expects pairs of [query, document_content]
        pairs = [[original_query, chunk.get('content', '')] for chunk in chunks]
        
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        # Combine chunks with their scores and sort
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the chunks, now in order of relevance
        return [chunk for score, chunk in scored_chunks]

    def _fetch_documents_by_ids(self, ids: List[str]) -> List[Dict]:
        """
        Fetches full document data (content + metadata) from ChromaDB for a list of IDs.
        """
        if not ids:
            return []
        
        # Fetch from Chroma, which returns a dictionary-like object
        results = self.collection.get(ids=ids, include=["metadatas", "documents"])
        
        # Reconstruct the original chunk format
        fetched_chunks = []
        for i, doc_id in enumerate(results['ids']):
            chunk = results['metadatas'][i] # Start with metadata
            chunk['content'] = results['documents'][i]
            chunk['chunk_id'] = doc_id
            fetched_chunks.append(chunk)
            
        return fetched_chunks

    def retrieve(
        self,
        original_query: str,
        year_filter: Optional[str] = None,
        k_semantic: int = 15,
        top_n_final: int = 5,
        use_query_expansion: bool = True,
        use_reranker: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline.

        Args:
            original_query: The user's question.
            year_filter: Optional year (e.g., "2023") to filter the search.
            k_semantic: Number of initial candidates to retrieve per query.
            top_n_final: The final number of documents to return after re-ranking.
            use_query_expansion: Whether to use the query expansion module.
            use_reranker: Whether to use the re-ranking module.
        """
        logger.info(f"Starting retrieval for query: '{original_query}'")
        
        # --- 1. Query Expansion (Optional) ---
        if use_query_expansion:
            queries = self.query_expander.expand(original_query)
            logger.debug(f"Expanded to {len(queries)} queries: {queries}")
        else:
            queries = [original_query]

        # --- 2. Build Metadata Filter (Optional) ---
        where_clause = {}
        if year_filter:
            where_clause['year'] = year_filter
            logger.info(f"Applying filter: year = {year_filter}")

        # --- 3. Semantic Search for each query ---
        fused_ids = set()
        query_embeddings = self.embedding_model.encode(queries, normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=k_semantic,
            where=where_clause if where_clause else None
        )
        
        for id_list in results['ids']:
            fused_ids.update(id_list)
            
        logger.info(f"Retrieved {len(fused_ids)} unique candidate documents from vector search.")

        if not fused_ids:
            return []

        # --- 4. Fetch Full Candidate Documents ---
        # This replaces the need for the in-memory document_store
        candidate_chunks = self._fetch_documents_by_ids(list(fused_ids))

        # --- 5. Re-ranking (Optional) ---
        if use_reranker:
            logger.info("Applying re-ranker...")
            final_chunks = self._rerank_documents(original_query, candidate_chunks)
        else:
            final_chunks = candidate_chunks
        
        # --- 6. Return Top N Results ---
        top_results = final_chunks[:top_n_final]
        logger.success(f"Retrieval complete. Returning {len(top_results)} final documents.")
        return top_results