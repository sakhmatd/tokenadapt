
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

def cache_embeddings(embed_model, embed_tokenizer, tokens: list, device: str, cache: dict) -> dict:
    """Cache embeddings for given tokens, updating the cache if necessary.

    Args:
        embed_model: Model to generate embeddings.
        embed_tokenizer: Tokenizer for the embedding model.
        tokens: List of tokens to cache.
        device: Device for embedding computation.
        cache: Existing cache dictionary.

    Returns:
        Updated cache dictionary.
    """
    for token in tqdm(tokens, desc="Caching embeddings"):
        if token not in cache:
            with torch.no_grad():
                inputs = embed_tokenizer(token, return_tensors="pt").to(device)
                embed = embed_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().tolist()
                cache[token] = embed
    return cache