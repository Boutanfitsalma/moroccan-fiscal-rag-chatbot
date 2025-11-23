# src/data_loader.py
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from loguru import logger

def _extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extracts the year from the filename and sets a static source.
    """
    metadata = {
        'year': 'unknown',
        'source': 'Note Circulaire'  # Hardcoded as requested
    }
    
    # Improved Regex: Finds the first 4-digit number starting with "20"
    # This correctly handles '2020part2,3.json'
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        metadata['year'] = year_match.group(1)
    else:
        logger.warning(f"Could not extract year from filename: '{filename}'. Defaulting to 'unknown'.")
    
    return metadata

def _ensure_unique_ids(chunks: List[Dict]) -> List[Dict]:
    """
    Checks for duplicate chunk_ids and makes them unique by appending a suffix.
    """
    id_counts = defaultdict(int)
    total_duplicates_found = 0
    
    # First pass: count all occurrences of each ID
    for chunk in chunks:
        id_counts[chunk['chunk_id']] += 1

    # Second pass: apply new IDs to duplicates
    processed_ids = defaultdict(int)
    for chunk in chunks:
        original_id = chunk['chunk_id']
        # If an ID appears more than once, it's a duplicate
        if id_counts[original_id] > 1:
            processed_count = processed_ids[original_id]
            # The first time we see a duplicate, we keep its original ID
            if processed_count > 0:
                new_id = f"{original_id}_{processed_count}"
                chunk['chunk_id'] = new_id
                if processed_count == 1: # Log only when the first duplicate is found
                    total_duplicates_found += 1
                    logger.warning(f"Duplicate ID '{original_id}' found. Renaming subsequent occurrences.")
            
            processed_ids[original_id] += 1
            
    if total_duplicates_found > 0:
        logger.success(f"Resolved {total_duplicates_found} sets of duplicate IDs.")
        
    return chunks

def load_and_prepare_chunks(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads, prepares, and ensures unique IDs for all chunks from JSON files.
    """
    all_chunks = []
    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in directory: {data_dir}")
        return []

    logger.info(f"Found {len(json_files)} JSON files to process in '{data_dir}'.")

    for file_path in json_files:
        logger.debug(f"Processing file: {file_path.name}")
        file_meta = _extract_metadata_from_filename(file_path.name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning(f"File {file_path.name} does not contain a list of chunks. Skipping.")
                    continue
                
                for i, chunk in enumerate(data):
                    # 1. Normalize ID field (robustly)
                    found_id = None
                    if 'chunk_id' in chunk:
                        found_id = chunk['chunk_id']
                    elif 'id' in chunk:
                        found_id = chunk.pop('id')
                    elif 'metadata' in chunk and isinstance(chunk.get('metadata'), dict) and 'id' in chunk['metadata']:
                        found_id = chunk['metadata'].pop('id')

                    if found_id:
                        chunk['chunk_id'] = str(found_id)
                    else:
                        logger.warning(f"Chunk #{i} in {file_path.name} is missing an ID. Skipping.")
                        continue
                    
                    
                    # 2. Prepare content for embedding, including table data
                    original_content = chunk.get('content', '') or chunk.get('article_title', '')
                    
                    # Flatten the nested 'metadata' dictionary first to access 'tables' easily
                    if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                        for meta_key, meta_val in chunk['metadata'].items():
                            if meta_key not in chunk:
                                chunk[meta_key] = meta_val
                        del chunk['metadata']

                    # Now, check for tables at the top level of the chunk
                    table_texts = []
                    if chunk.get('has_tables') and 'tables' in chunk:
                        for table in chunk.get('tables', []):
                            if 'as_text' in table:
                                table_texts.append(table['as_text'])

                    # Create the final, enriched content string for vectorization
                    if table_texts:
                        # Combine original text with a clear separator and all table texts
                        enriched_content = original_content + "\n\n--- Table Data ---\n" + "\n".join(table_texts)
                    else:
                        enriched_content = original_content
                    
                    # We replace the original 'content' key with our new enriched version
                    chunk['content'] = enriched_content
                    
                    if not chunk['content']:
                        logger.warning(f"Chunk {chunk['chunk_id']} has no final content. Skipping.")
                        continue

                    # 3. Inject file-level metadata
                    chunk['source_file'] = file_path.name
                    chunk.update(file_meta) # Adds 'year' and 'source'
                    
                    # 4. Flatten the nested 'metadata' dictionary
                    if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                        for meta_key, meta_val in chunk['metadata'].items():
                            if meta_key not in chunk:
                                chunk[meta_key] = meta_val
                        del chunk['metadata']

                    all_chunks.append(chunk)

            except Exception as e:
                logger.error(f"An error occurred while processing {file_path.name}: {e}")
                
    logger.success(f"Successfully loaded and parsed {len(all_chunks)} chunks from all files.")
    
    # 5. Final step: Ensure all chunk IDs are unique across the entire dataset
    if all_chunks:
        logger.info("Checking for duplicate chunk IDs across all loaded files...")
        all_chunks = _ensure_unique_ids(all_chunks)

    return all_chunks