from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import display_markdown

# Define model and tokenizer paths
model_path = "Qwen/Qwen2.5-1.5B"  # Replace with your LLM path
new_tokenizer_path = "fhai50032/QTK-81K"  # Replace with your new tokenizer path

# **Load the pre-trained LLM on CPU**
# Using bfloat16 for memory efficiency; loads on CPU by default unless moved
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cpu")

# **Load tokenizers**
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Original tokenizer
new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path)  # New tokenizer

# **Load Indic-BERT on CUDA** for embedding computations
bert_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v2-moe").to("cuda")
bert_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")

# **Create vocabulary mappings**
old_id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
old_token_to_id = tokenizer.get_vocab()
new_id_to_token = {v: k for k, v in new_tokenizer.get_vocab().items()}
new_token_to_id = new_tokenizer.get_vocab()

# **Identify common tokens**
common_vocab = list(set(new_token_to_id.keys()) & set(old_token_to_id.keys()))
print(f"Tokens to be Transplanted: {len(common_vocab)}")
print(f"Tokens to be Initialized: {len(new_tokenizer) - len(common_vocab)}")

# **Define new tokens** (present in new tokenizer but not in old)
new_tokens = set(new_token_to_id.keys()) - set(common_vocab)

with torch.no_grad():
    # **Initialize new embedding matrix on CPU**
    new_input_embed = torch.rand(
        (len(new_tokenizer), model.get_input_embeddings().weight.shape[1]),
        dtype=torch.bfloat16,
        device="cpu"
    )
    new_input_embed.normal_(
        mean=model.get_input_embeddings().weight.mean().item(),
        std=model.get_input_embeddings().weight.std().item()
    )
    
    # **Copy embeddings for common tokens** (on CPU)
    for token in common_vocab:
        old_id = old_token_to_id[token]
        new_id = new_token_to_id[token]
        new_input_embed[new_id] = model.get_input_embeddings().weight[old_id].clone()
    
    for new_token_str in tqdm(new_tokens, desc="Initializing new tokens"):
        new_id = new_token_to_id[new_token_str]
        token_humanish = new_tokenizer.decode([new_id])  # Convert token ID to human-readable form
        old_token_ids = tokenizer.encode(token_humanish, add_special_tokens=False)
        if not old_token_ids:
            continue  # Skip if no valid subtokens are found
        token_humanish_base = [tokenizer.decode([x]) for x in old_token_ids]
        
        with torch.no_grad():
            inputs_full = bert_tokenizer(token_humanish, return_tensors="pt").to("cuda")
            embed_full = bert_model(**inputs_full).last_hidden_state.mean(dim=1).squeeze(0).cpu()
        
        sub_embeds = []
        for subtoken in token_humanish_base:
            inputs_sub = bert_tokenizer(subtoken, return_tensors="pt").to("cuda")
            sub_embed = bert_model(**inputs_sub).last_hidden_state.mean(dim=1).squeeze(0).cpu()
            sub_embeds.append(sub_embed)
        sub_embeds = torch.stack(sub_embeds)  # [num_subtokens, hidden_size] on CPU
        
        similarities = F.cosine_similarity(embed_full.unsqueeze(0), sub_embeds, dim=1)

        weights = F.softmax(similarities, dim=0)
        old_embeds = torch.stack([model.get_input_embeddings().weight[old_id] for old_id in old_token_ids])
        weighted_sum = (weights.unsqueeze(1) * old_embeds).sum(dim=0)
        new_input_embed[new_id] = weighted_sum
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.resize_token_embeddings(len(new_tokenizer))
    
    # **Assign new embeddings** to input and output layers (weights are tied)
    model.get_input_embeddings().weight.copy_(new_input_embed)
    model.get_output_embeddings().weight.copy_(new_input_embed)

bert_model.to("cpu")
torch.cuda.empty_cache()  # Clear GPU memory

model.to("cuda")

test_input = "कृषि"  # Replace with a suitable test string
inputs = new_tokenizer([test_input], return_tensors="pt").to("cuda")
outputs = model.generate(inputs["input_ids"], max_new_tokens=128)
display_markdown(new_tokenizer.batch_decode(outputs)[0], raw=True)

print("Model adaptation complete!")
