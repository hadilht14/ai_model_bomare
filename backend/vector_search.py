# backend/vector_search.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os # Added for robust path

# --- Initialization ---
MODEL_NAME = 'all-MiniLM-L6-v2'
log = logging.getLogger(__name__)

log.info(f"Loading sentence transformer model: {MODEL_NAME}")
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    log.error(f"Failed to load Sentence Transformer model '{MODEL_NAME}': {e}", exc_info=True)
    # This should ideally stop the application or be handled gracefully
    raise RuntimeError(f"Failed to initialize sentence transformer model: {MODEL_NAME}") from e

# --- Load Data & Flatten ---
# json_path will be an absolute path passed from chatbot_core.py
def load_data(json_path: str):
    """Loads troubleshooting data from a JSON file and flattens it."""
    # ... (rest of your load_data function is identical) ...
    try:
        with open(json_path, "r", encoding='utf-8') as f:
            raw_data = json.load(f) # This should be a list of TV model entries
        if not isinstance(raw_data, list):
            log.error(f"Error: Data in {json_path} is not a valid JSON list of TV model entries.")
            return None

        flattened_data = []
        for model_entry in raw_data:
            if not isinstance(model_entry, dict):
                log.warning(f"Skipping non-dict item in raw_data: {str(model_entry)[:100]}")
                continue

            model_name = model_entry.get("model")
            troubleshooting_issues = model_entry.get("troubleshooting_issues")
            model_general_images = model_entry.get("images")

            if not model_name:
                log.warning(f"Skipping model entry due to missing 'model' field: {str(model_entry)[:100]}")
                continue

            if not isinstance(troubleshooting_issues, list):
                log.warning(f"Model '{model_name}' has no 'troubleshooting_issues' list or it's invalid. Skipping its issues for RAG indexing.")
                continue

            for issue_detail in troubleshooting_issues:
                if not isinstance(issue_detail, dict):
                    log.warning(f"Skipping non-dict item in troubleshooting_issues for model '{model_name}': {str(issue_detail)[:100]}")
                    continue

                issue_text = issue_detail.get("issue")
                steps = issue_detail.get("steps")

                if not issue_text or not isinstance(issue_text, str) or not issue_text.strip():
                    log.warning(f"Skipping issue for model '{model_name}' due to missing, empty, or invalid 'issue' text: {str(issue_detail)[:100]}")
                    continue

                flattened_entry = {
                    "model": model_name,
                    "issue": issue_text.strip(),
                    "steps": steps if isinstance(steps, list) else [],
                    "images": model_general_images if isinstance(model_general_images, dict) else None
                }
                flattened_data.append(flattened_entry)

        if not flattened_data:
            log.error(f"No valid issues to index were extracted and flattened from {json_path}")
            return None

        log.info(f"Successfully loaded and flattened {len(flattened_data)} issue items from {len(raw_data)} model entries in {json_path}")
        return flattened_data
    except FileNotFoundError:
        log.error(f"Error: Data file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"Error: Could not decode JSON from {json_path}. Error: {e}")
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred loading and flattening data: {e}", exc_info=True)
        return None
# --- Index Creation (create_faiss_index) ---
# ... (identical to your version) ...
def create_faiss_index(data: list): # data is now the FLATTENED list
    """Creates a FAISS index for the 'issue' field in the (flattened) data."""
    if not data:
        log.error("Cannot create index from empty or invalid (flattened) data.")
        return None, None

    texts_to_embed = []
    text_to_original_data_idx_map = [] # This will map to indices in the FLATTENED data list

    log.info("Preparing 'issue' texts from flattened data for embedding...")
    valid_items_count = 0
    for i, item in enumerate(data): # item is a dict from the flattened_data
        if isinstance(item, dict) and "issue" in item and isinstance(item["issue"], str) and item["issue"].strip():
            texts_to_embed.append(item["issue"]) # item["issue"] is directly accessible
            text_to_original_data_idx_map.append(i) # Store index from the flattened list
            valid_items_count += 1
        else:
            log.warning(f"Flattened item at index {i} is invalid or missing 'issue' field. Skipping. Item: {str(item)[:100]}")

    if not texts_to_embed:
        log.error("No valid 'issue' fields found in flattened data to index.")
        return None, None

    log.info(f"Found {valid_items_count} valid issues from flattened data to index.")
    try:
        log.info(f"Encoding {len(texts_to_embed)} issues using '{MODEL_NAME}'...")
        embeddings = model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True) # show_progress_bar=False for server
        if embeddings is None or embeddings.size == 0:
            log.error("Encoding resulted in empty embeddings array.")
            return None, None
        embeddings = embeddings.astype('float32')
        dimension = embeddings.shape[1]
        log.info(f"Embeddings created with dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        log.info(f"FAISS index created successfully with {index.ntotal} vectors.")
        return index, text_to_original_data_idx_map
    except Exception as e:
        log.error(f"Error creating FAISS index: {e}", exc_info=True)
        return None, None
# --- Strict Search Function (search_relevant_guides) ---
# ... (identical to your version) ...
def search_relevant_guides(query_text: str, target_model: str, data: list, index: faiss.Index, text_to_original_data_idx_map: list, k_results: int = 10):
    """
    Searches for relevant guides from the FLATTENED data,
    STRICTLY matching the target_model.
    Returns a single guide dictionary (from flattened_data) or None.
    """
    if not all([index, data, text_to_original_data_idx_map, query_text, target_model]):
        log.error("Search cannot be performed - missing index, data (flattened), map, query, or target model.")
        return None
    if index.ntotal == 0:
        log.warning("FAISS index is empty. Cannot perform search.")
        return None
    try:
        log.info(f"Encoding search query: '{query_text}'")
        query_embedding = model.encode([query_text], convert_to_numpy=True)
        if query_embedding is None or query_embedding.size == 0:
             log.error("Failed to encode search query.")
             return None
        query_embedding_np = query_embedding.astype('float32')
        effective_k = min(k_results, index.ntotal, len(data)) 

        log.info(f"Searching FAISS index for top {effective_k} matches for query '{query_text[:60]}...'")
        distances, embedding_indices = index.search(query_embedding_np, k=effective_k)
        
        matched_flattened_guides_info = {} 

        for i, emb_idx in enumerate(embedding_indices[0]): 
            if emb_idx < 0 or emb_idx >= len(text_to_original_data_idx_map):
                 log.warning(f"FAISS returned invalid embedding index {emb_idx}. Skipping.")
                 continue
            flattened_data_idx = text_to_original_data_idx_map[emb_idx] 
            if flattened_data_idx < 0 or flattened_data_idx >= len(data):
                 log.warning(f"Mapped flattened_data_idx {flattened_data_idx} is out of bounds. Skipping.")
                 continue

            score = float(distances[0][i]) 
            rank = i + 1 

            if flattened_data_idx not in matched_flattened_guides_info or score < matched_flattened_guides_info[flattened_data_idx]['score']:
                 matched_flattened_guides_info[flattened_data_idx] = {'score': score, 'rank': rank}

        sorted_unique_flattened_indices = sorted(
            matched_flattened_guides_info.keys(),
            key=lambda idx: matched_flattened_guides_info[idx]['rank']
        )

        log.debug(f"Top {len(sorted_unique_flattened_indices)} unique flattened guide candidates (sorted by FAISS rank): {sorted_unique_flattened_indices}")

        found_guide_dict = None
        target_model_lower = target_model.strip().lower()

        for flattened_idx in sorted_unique_flattened_indices:
            candidate_guide = data[flattened_idx] 

            if not isinstance(candidate_guide, dict):
                 log.warning(f"Data item at flattened_idx {flattened_idx} is not a dictionary. Skipping.")
                 continue

            candidate_model = candidate_guide.get("model", "").strip().lower() 
            candidate_issue = candidate_guide.get("issue","N/A")

            rank_info = matched_flattened_guides_info[flattened_idx]
            log.debug(f"Checking candidate (FlatIdx {flattened_idx}, FAISS Rank {rank_info['rank']}, Score {rank_info['score']:.4f}): Model '{candidate_model}' vs Target '{target_model_lower}'. Issue: '{candidate_issue[:60]}...'")

            if candidate_model == target_model_lower:
                log.info(f"*** SUCCESS: Found guide matching target model '{target_model}' (FlatIdx {flattened_idx}) at rank {rank_info['rank']}. Issue: '{candidate_issue}' ***")
                found_guide_dict = candidate_guide
                break 

        if not found_guide_dict:
             log.warning(f"No guide strictly matching target model '{target_model}' found within top {effective_k} semantic matches for query '{query_text[:60]}...'.")

        return found_guide_dict

    except faiss.FaissException as e: 
        log.error(f"FAISS Error during search: {e}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Unexpected error during search: {e}", exc_info=True)
        return None