import os
import tempfile
import json
import asyncio
from typing import List, Generator
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
import sys

# --- Configure Logger ---
logger.remove()
logger.add(sys.stdout, colorize=True, format="<level>{level: <8}</level> | <level>{message}</level>")

# --- Import RAG Components ---
from src.config import MODEL_PAIRS, LOCAL_MODELS, DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

# --- Lazy Loading Components to avoid initialization conflicts ---
INDEXER = None
CHUNKER = None
PIPELINE_INITIALIZED = False

def initialize_rag_components():
    """Initialize RAG components lazily when first needed."""
    global INDEXER, CHUNKER, PIPELINE_INITIALIZED
    
    if PIPELINE_INITIALIZED:
        logger.info("RAG components already initialized.")
        return
    
    logger.info("Initializing RAG backend components...")
    
    try:
        # Import here to avoid circular imports and early initialization
        from src.indexer import VectorIndexer
        from src.chunker import FiscalNoteChunker
        
        logger.info("Creating VectorIndexer...")
        INDEXER = VectorIndexer(db_path=DB_PATH, collection_name=COLLECTION_NAME, model_name=EMBEDDING_MODEL_NAME, force_reindex=False)
        
        logger.info("Creating FiscalNoteChunker...")
        CHUNKER = FiscalNoteChunker()
        
        PIPELINE_INITIALIZED = True
        logger.success("RAG backend components initialized successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise e

# --- API Definition ---
app = FastAPI(
    title="Fiscal RAG API", 
    version="1.0.0",
    # Ensure proper UTF-8 encoding handling
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS and Authentication Middleware ---
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI-Compatible Models for the UI ---
class ModelCard(BaseModel): id: str; object: str = "model"; owned_by: str = "user"
class ModelList(BaseModel): object: str = "list"; data: List[ModelCard]
class Message(BaseModel): 
    role: str
    content: str
    
    class Config:
        str_to_lower = False
        
class ChatCompletionRequest(BaseModel): 
    model: str
    messages: List[Message]
    stream: bool = False
    
    class Config:
        str_to_lower = False

# --- Helper to map model names for the UI ---
def get_model_choices():
    choices = {}
    for pair_data in MODEL_PAIRS.values():
        choices[f"OpenRouter: {pair_data['classic']['name']}"] = f"openrouter/{pair_data['classic']['id']}"
        choices[f"OpenRouter: {pair_data['reasoning']['name']}"] = f"openrouter/{pair_data['reasoning']['id']}"
    for model_data in LOCAL_MODELS.values():
        choices[f"Local: {model_data['name']}"] = f"ollama/{model_data['id']}"
    return choices

# --- Streaming Generator for Chat ---
async def stream_rag_response(query: str, provider: str, model_id: str):
    """Streams the RAG response in the format Open WebUI expects."""
    initialize_rag_components()
    from src.pipeline import run_rag_pipeline
    response_content = run_rag_pipeline(query=query, provider=provider, model_id=model_id)
    
    for chunk in response_content.split():
        json_data = json.dumps({
            "id": "chatcmpl-stream", "object": "chat.completion.chunk", "model": model_id,
            "choices": [{"delta": {"content": " " + chunk}}]
        })
        yield f"data: {json_data}\n\n"
        await asyncio.sleep(0.02)
    yield "data: [DONE]\n\n"

# --- API Endpoints ---
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Provides a list of available models for Open WebUI."""
    model_cards = [ModelCard(id=display_name) for display_name in get_model_choices().keys()]
    return ModelList(data=model_cards)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Main chat endpoint, compatible with OpenAI and supporting streaming."""
    user_query = request.messages[-1].content
    
    # Debug: Log the received query to check encoding
    logger.info(f"Received query: {repr(user_query)}")
    logger.info(f"Query type: {type(user_query)}")
    
    # Ensure proper UTF-8 handling
    if isinstance(user_query, bytes):
        user_query = user_query.decode('utf-8')
    
    logger.info(f"Processed query: {user_query}")
    logger.debug(f"Received query: {repr(user_query)}")
    logger.debug(f"Query length: {len(user_query)}")
    
    model_map = get_model_choices()
    provider_model_id = model_map.get(request.model, f"openrouter/{MODEL_PAIRS['Duel Qwen']['classic']['id']}") # Fallback to Qwen Classic
    provider, model_id = provider_model_id.split('/', 1)

    if request.stream:
        return StreamingResponse(stream_rag_response(user_query, provider, model_id), media_type="text/event-stream")
    else:
        initialize_rag_components()
        from src.pipeline import run_rag_pipeline
        response = run_rag_pipeline(query=user_query, provider=provider, model_id=model_id)
        return {"choices": [{"message": {"role": "assistant", "content": response}}]}

# THIS IS THE NEW CUSTOM ENDPOINT FOR UPLOADING
@app.post("/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Handles uploading, chunking, and indexing of a new PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Initialize RAG components if not already done
    initialize_rag_components()
    
    # Debug: Check if CHUNKER is properly initialized
    logger.info(f"CHUNKER status: {CHUNKER is not None}")
    if CHUNKER is None:
        logger.error("CHUNKER is still None after initialization!")
        raise HTTPException(status_code=500, detail="Failed to initialize CHUNKER component")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            pdf_path = tmp.name

        logger.info(f"Starting processing for uploaded file: {file.filename}")
        
        # 1. Chunk the document using your script
        pages_data, stats = CHUNKER.extract_text_and_tables_from_pdf_with_pages(pdf_path)
        if not stats.get("success"):
            raise ValueError(f"Failed to extract text from PDF. Details: {stats.get('error', 'Unknown')}")
        
        new_chunks = CHUNKER.process_document(pages_data)
        if not new_chunks:
            raise ValueError("Chunking script produced no chunks.")
            
        # 2. Prepare chunks (this logic could be moved to a helper function)
        prepared_chunks = []
        for chunk in new_chunks:
            if 'metadata' in chunk:
                for k, v in chunk['metadata'].items():
                    if k not in chunk: chunk[k] = v
                del chunk['metadata']
            prepared_chunks.append(chunk)

        # 3. Save processed chunks as JSON in data folder
        from datetime import datetime
        import json
        from pathlib import Path
        
        # Generate filename based on original filename and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(file.filename).stem
        json_filename = f"{base_name}_{timestamp}.json"
        json_path = Path("data") / json_filename
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Convert chunks to the same format as existing JSON files
        json_chunks = []
        for chunk in prepared_chunks:
            json_chunk = {
                "content": chunk.get("content", ""),
                "metadata": {
                    "type": chunk.get("type", "uploaded_document"),
                    "page_start": chunk.get("page_number", 1),
                    "page_end": chunk.get("end_page_number", chunk.get("page_number", 1)),
                    "id": chunk.get("chunk_id", f"uploaded_{timestamp}"),
                    "source_file": json_filename,
                    "original_filename": file.filename,
                    "upload_timestamp": timestamp,
                    "document_title": chunk.get("document_title", ""),
                    "section_hierarchy": chunk.get("section_hierarchy", ""),
                    "sub_topic": chunk.get("sub_topic", ""),
                    "main_topic": chunk.get("main_topic", "")
                },
                "length": len(chunk.get("content", "")),
                "word_count": len(chunk.get("content", "").split())
            }
            json_chunks.append(json_chunk)
        
        # Save JSON file to data folder
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed chunks to: {json_path}")
        
        # 4. Ensure unique IDs and index
        initialize_rag_components()
        from src.data_loader import _ensure_unique_ids
        unique_chunks = _ensure_unique_ids(prepared_chunks)
        INDEXER.index(unique_chunks)
        
        success_msg = f"Successfully processed and indexed {len(unique_chunks)} chunks from {file.filename}. JSON saved as {json_filename}."
        logger.success(success_msg)
        return {"status": "success", "message": success_msg, "json_file": json_filename, "chunks_count": len(unique_chunks)}
        
    except Exception as e:
        logger.error(f"Error during file processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker container monitoring."""
    return {"status": "healthy", "service": "Moroccan Fiscal RAG API"}

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Moroccan Fiscal RAG API server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        access_log=True
    )