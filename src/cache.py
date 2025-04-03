
# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script is part of the Tokenizer Transplantation Tool.



import json
import torch
from tqdm.auto import tqdm

def load_cache(cache_file: str) -> dict:
    """Load cached embeddings from a JSON file.

    Args:
        cache_file: Path to the cache file.

    Returns:
        Dictionary of cached embeddings.
    """
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache_file: str, cache: dict) -> None:
    """Save cached embeddings to a JSON file.

    Args:
        cache_file: Path to the cache file.
        cache: Dictionary of embeddings to save.
    """
    with open(cache_file, "w") as f:
        json.dump(cache, f)


def mean_pooling(model_output, attention_mask): 
    """Performs mean pooling on the token embeddings, considering the attention mask."""
    token_embeddings = model_output.last_hidden_state 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cache_embeddings(embed_model, embed_tokenizer, tokens: list, device: str, cache: dict, batch_size: int = 4) -> dict:
    """
    Cache embeddings for given tokens using batch processing, updating the cache if necessary.

    Args:
        embed_model: Model to generate embeddings.
        embed_tokenizer: Tokenizer for the embedding model.
        tokens: List of unique tokens (strings) to potentially cache.
        device: Device for embedding computation ('cuda', 'cpu', etc.).
        cache: Existing cache dictionary (token -> embedding_list).
        batch_size: Number of tokens to process in each batch.

    Returns:
        Updated cache dictionary.
    """
    tokens_to_process = sorted(list(set(token for token in tokens if token not in cache))) # Ensure uniqueness and determinism

    if not tokens_to_process:
        print("All required tokens are already cached.")
        return cache

    print(f"Found {len(tokens_to_process)} unique tokens needing embedding.")

    
    for i in tqdm(range(0, len(tokens_to_process), batch_size), desc="Caching embeddings (batched)"):
        batch_tokens = tokens_to_process[i : i + batch_size]

        
        inputs = embed_tokenizer(
            batch_tokens,
            padding=True,         
            truncation=True,     
            return_tensors="pt",  
            max_length=embed_tokenizer.model_max_length,
        ).to(device)

        
        with torch.no_grad():
            outputs = embed_model(**inputs)

        batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])

        
        batch_embeddings_list = batch_embeddings.cpu().tolist()

        for token, embedding in zip(batch_tokens, batch_embeddings_list):
            cache[token] = embedding

    return cache