# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script is part of the Tokenizer Transplantation Tool.
# It orchestrates the transplantation process, determining whether embeddings are tied or untied,
# and calls the appropriate transplantation function from tied.py or untied.py.
# It also handles caching of embeddings for full tokens and subtokens using cache.py.

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from tied import transplant_tied_embeddings
from untied import transplant_untied_embeddings
from cache import load_cache, save_cache, cache_embeddings

def main(args):
    """Main function to execute the tokenizer transplantation process."""
    # --------------- Setup ------------------
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    print(f"Data type selected: {args.dtype}")

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")

    # --------------- Loading Models and Tokenizers ---------------
    
    print("Loading pre-trained model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cpu", token=args.hf_token
    )
    
    old_generation_config = model.generation_config

    print("Loading tokenizers...")
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.hf_token)
    new_tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer_path, token=args.hf_token)

    
    print("Loading embedding model...")
    embed_model = AutoModel.from_pretrained(args.embedding_model_path, trust_remote_code=True).to(device)
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path, trust_remote_code=True)

    # --------------- Transplant Start Phase 1 -------------------------

    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    shared_vocab = list(set(new_vocab.keys()) & set(old_vocab.keys()))
    unique_tokens = set(new_vocab.keys()) - set(shared_vocab)
    print(f"Shared tokens: {len(shared_vocab)}")
    print(f"Unique tokens to initialize: {len(unique_tokens)}")

    
    embed_model_name = args.embedding_model_path.split("/")[-1]
    cache_file = f"cache_{embed_model_name}.json"

    
    cache = load_cache(cache_file)

    
    full_tokens = [new_tokenizer.decode([new_vocab[token_str]]) for token_str in unique_tokens]
    cache = cache_embeddings(embed_model, embed_tokenizer, full_tokens, device, cache)

    
    subtokens = []
    for token_str in unique_tokens:
        full_token = new_tokenizer.decode([new_vocab[token_str]])
        old_ids = old_tokenizer.encode(full_token, add_special_tokens=False)
        subtokens.extend(old_tokenizer.decode([oid]) for oid in old_ids)
    subtokens = list(set(subtokens))  
    cache = cache_embeddings(embed_model, embed_tokenizer, subtokens, device, cache)

    
    save_cache(cache_file, cache)

    
    tied = getattr(model.config, "tie_word_embeddings", False)
    if not tied:
        input_embeds = model.get_input_embeddings().weight
        output_embeds = model.get_output_embeddings()
        tied = output_embeds is None or input_embeds is output_embeds.weight
    print(f"Tied embeddings detected: {tied}")

    # --------------- Transplant Phase 2 -------------------------

    if tied:
        transplant_tied_embeddings(
            model, new_tokenizer, shared_vocab, unique_tokens, cache, cache,
            old_vocab, new_vocab, old_tokenizer, dtype ,args.temperature, args.multiple_of
        )
    else:
        transplant_untied_embeddings(
            model, new_tokenizer, shared_vocab, unique_tokens, cache, cache,
            old_vocab, new_vocab, old_tokenizer, dtype,args.temperature , args.multiple_of
        )

    # ------------- Clean-Up -----------------------
    try:
        eos_id = getattr(new_tokenizer, "eos_token_id", getattr(new_tokenizer, "bos_token_id", None))
        bos_id = getattr(new_tokenizer, "bos_token_id", getattr(new_tokenizer, "eos_token_id", None))
        pad_id = getattr(new_tokenizer, "pad_token_id", getattr(new_tokenizer, "eos_token_id", None))
        model.config.update({"pad_token_id": pad_id, "eos_token_id": eos_id, "bos_token_id": bos_id})
        model.generation_config = old_generation_config
        model.generation_config.update({"pad_token_id": pad_id, "eos_token_id": eos_id, "bos_token_id": bos_id})
        new_tokenizer.chat_template=old_tokenizer.chat_template
    except Exception as e:
        print(f"Config update failed: {e}")

    
    print(f"Saving to Hugging Face as {args.new_model_name}...")
    model.push_to_hub(args.new_model_name, private=False, token=args.hf_token)
    new_tokenizer.push_to_hub(args.new_model_name, private=False, token=args.hf_token)
    print("Transplantation completed!")

#  ------------- End ------------------ 

if __name__ == "__main__":
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end
        def __eq__(self, other):
            return self.start <= other <= self.end
    
    parser = argparse.ArgumentParser(description="Tokenizer Transplantation ")
    parser.add_argument(
        "-model", "--model_path", required=True, help="Path to the original model"
    )
    parser.add_argument(
        "-tk", "--new_tokenizer_path", required=True, help="Path to the new tokenizer"
    )
    parser.add_argument(
        "-embed", "--embedding_model_path", default="nomic-ai/nomic-embed-text-v2-moe",
        help="Path to embedding model; defaults to nomic-ai/nomic-embed-text-v2-moe"
    )
    parser.add_argument(
        "-repo", "--new_model_name", required=True, help="HF's Repo name for the new model"
    )
    parser.add_argument(
        "-auth", "--hf_token", required=True, help="Hugging Face authentication token"
    )
    parser.add_argument(
        "-temp", "--temperature", default=0.3, 
        help="Temprature for more expresive weighting 0.3 is default more than this is more bland ; less than this is more expressive", 
        type=float,choices=[Range(0.0, 1.0)]
    )
    parser.add_argument(
        "-pad","--multiple_of" , default = 128,
        help="When Resizing model ; will resize to a multiple of earlier papers proved padding to power of 2 helps in throughput; default is 128",
        type=int
    )
    parser.add_argument(
        "-d", "--dtype", default="fp32", choices=["bf16", "fp16", "fp32"],
        help="Model and Processing data type, default : fp32"
    )
    args = parser.parse_args()
    main(args)