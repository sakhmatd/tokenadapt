
# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script is part of the Tokenizer Transplantation Tool.

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import math
from heuristics import calculate_local_embedding, calculate_global_embedding
import faiss 
from typing import Optional 

def transplant_tied_embeddings(
    model, new_tokenizer: AutoTokenizer, shared_vocab: list, unique_tokens: set,
    full_token_embeds_cache: dict, subtoken_embeds_cache: dict, old_vocab: dict,
    new_vocab: dict, old_tokenizer: AutoTokenizer, data_type: torch.dtype,
    temperature: float, pad_to_multiple_of: int,
    faiss_index: Optional[faiss.Index], index_to_token: Optional[dict], k: int, global_weight: float
    ) -> None:
    """
    Transplants embeddings for a model with tied input/output embeddings.
    Uses heuristic helpers, calculating weights once.
    """

    eps = 1e-5
    calc_temperature = temperature + eps

    try:
        calc_device = model.device if model.device.type != 'meta' else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except AttributeError:
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for heuristic calculations: {calc_device}")


    with torch.no_grad():
        original_input_embeddings = model.get_input_embeddings().weight.clone()
        original_output_embeddings = None
        embed_dim = original_input_embeddings.shape[1]

        new_vocab_size = len(new_tokenizer)
        padded_size = math.ceil(new_vocab_size / pad_to_multiple_of) * pad_to_multiple_of
        new_embeddings = torch.empty(padded_size, embed_dim, dtype=data_type, device='cpu')
        mean = original_input_embeddings.mean().item()
        std = original_input_embeddings.std().item()
        new_embeddings.normal_(mean=mean, std=std)
        print(f"Initialized new embedding matrix with size {padded_size}x{embed_dim}")

        
        copied_count = 0
        for token in tqdm(shared_vocab, desc="Copying shared token embeddings"):
            old_id = old_vocab.get(token)
            new_id = new_vocab.get(token)
            if old_id is not None and new_id is not None and (0 <= old_id < original_input_embeddings.shape[0]):
                 new_embeddings[new_id] = original_input_embeddings[old_id].to(device='cpu', dtype=data_type)
                 copied_count += 1
        print(f"Copied {copied_count}/{len(shared_vocab)} shared token embeddings.")

        local_success = 0
        global_success = 0
        combined_success = 0
        random_init_count = 0

        local_weight = 1.0 - global_weight
        use_global = global_weight > 0 and faiss_index is not None
        use_local = local_weight > 0

        print(f"Initializing unique tokens (Tied). Global heuristic enabled: {use_global} (weight={global_weight:.2f}), Local heuristic enabled: {use_local} (weight={local_weight:.2f})")

        for token_str in tqdm(unique_tokens, desc="Initializing unique tokens (Tied Hybrid)"):
            new_id = new_vocab.get(token_str)
            if new_id is None: continue

            e_local_in = None 
            e_global_in = None

            # Calculate local 
            if use_local:
                e_local_in, _ = calculate_local_embedding(
                    token_str, new_id, new_tokenizer, old_tokenizer,
                    full_token_embeds_cache, subtoken_embeds_cache,
                    original_input_embeddings, original_output_embeddings, 
                    calc_temperature, data_type, calc_device
                )
                if e_local_in is not None: local_success += 1


            if use_global:
                full_token_decoded = new_tokenizer.decode([new_id])
                e_global_in, _ = calculate_global_embedding(
                    full_token_decoded, full_token_embeds_cache, faiss_index,
                    index_to_token, old_vocab,
                    original_input_embeddings, original_output_embeddings, 
                    k, calc_temperature, data_type, calc_device
                )
                if e_global_in is not None: global_success += 1

            final_embedding = None
            if e_local_in is not None and e_global_in is not None:
                final_embedding = (local_weight * e_local_in + global_weight * e_global_in).to(dtype=data_type)
                combined_success += 1
            elif e_local_in is not None:
                final_embedding = e_local_in.to(dtype=data_type)
            elif e_global_in is not None:
                final_embedding = e_global_in.to(dtype=data_type)

            if final_embedding is not None:
                new_embeddings[new_id] = final_embedding.cpu()
            else:
                random_init_count += 1

        print(f"Initialization complete for {len(unique_tokens)} unique tokens:")
        print(f"  - Local heuristic succeeded for: {local_success}")
        print(f"  - Global heuristic succeeded for: {global_success}")
        print(f"  - Combined successfully (both ran & succeeded): {combined_success}")
        print(f"  - Remained randomly initialized: {random_init_count}")

        print("Resizing model token embeddings...")
        for param in model.parameters(): param.requires_grad = False
        model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=pad_to_multiple_of)
        print(f"Model embedding size after resize: {model.get_input_embeddings().weight.shape}")

        target_device = model.get_input_embeddings().weight.device
        target_dtype = model.get_input_embeddings().weight.dtype
        new_embeddings_tensor = new_embeddings.to(target_device, dtype=target_dtype)

        if new_embeddings_tensor.shape == model.get_input_embeddings().weight.shape:
             model.get_input_embeddings().weight.copy_(new_embeddings_tensor)
        else:
             print(f"Error: Shape mismatch for input embeddings. Expected {model.get_input_embeddings().weight.shape}, got {new_embeddings_tensor.shape}.")
        model.tie_weights()

        print("Embedding update complete.")