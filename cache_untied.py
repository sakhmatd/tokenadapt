from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import display_markdown

model_path = "EleutherAI/pythia-1b"  # Replace with your LLM path
new_tokenizer_path = "fhai50032/QTK-81K"  # Replace with your new tokenizer path

# Using bfloat16 for memory efficiency; loads on CPU by default unless moved
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)  # Original tokenizer
new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path)  # New tokenizer

bert_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v2-moe").to("cuda")
bert_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")

old_id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
old_token_to_id = tokenizer.get_vocab()
new_id_to_token = {v: k for k, v in new_tokenizer.get_vocab().items()}
new_token_to_id = new_tokenizer.get_vocab()

common_vocab = list(set(new_token_to_id.keys()) & set(old_token_to_id.keys()))
print(f"Tokens to be Transplanted: {len(common_vocab)}")
print(f"Tokens to be Initialized: {len(new_tokenizer) - len(common_vocab)}")

new_tokens = set(new_token_to_id.keys()) - set(common_vocab)

bert_embedding_cache = {}

def get_bert_embedding(text):
    """Get BERT embedding for a given text, using cache to avoid redundant computations."""
    if text in bert_embedding_cache:
        return bert_embedding_cache[text]
    
    with torch.no_grad():
        inputs = bert_tokenizer(text, return_tensors="pt").to("cuda")
        embedding = bert_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu()
    
    bert_embedding_cache[text] = embedding
    return embedding

with torch.no_grad():
    new_input_embed = torch.rand(
        (len(new_tokenizer), model.get_input_embeddings().weight.shape[1]),
        dtype=torch.bfloat16,
        device="cpu"
    )
    new_output_embed = torch.rand(
        (len(new_tokenizer), model.get_output_embeddings().weight.shape[1]),
        dtype=torch.bfloat16,
        device="cpu"
    )
    
    new_input_embed.normal_(
        mean=model.get_input_embeddings().weight.mean().item(),
        std=model.get_input_embeddings().weight.std().item()
    )
    new_output_embed.normal_(
        mean=model.get_output_embeddings().weight.mean().item(),
        std=model.get_output_embeddings().weight.std().item()
    )
    
    for token in common_vocab:
        old_id = old_token_to_id[token]
        new_id = new_token_to_id[token]
        new_input_embed[new_id] = model.get_input_embeddings().weight[old_id].clone()
        new_output_embed[new_id] = model.get_output_embeddings().weight[old_id].clone()
    
    for new_token_str in tqdm(new_tokens, desc="Initializing new tokens"):
        new_id = new_token_to_id[new_token_str]
        token_humanish = new_tokenizer.decode([new_id])  # Convert token ID to human-readable form
        old_token_ids = tokenizer.encode(token_humanish, add_special_tokens=False)
        if not old_token_ids:
            continue  # Skip if no valid subtokens are found
        token_humanish_base = [tokenizer.decode([x]) for x in old_token_ids]
        
        embed_full = get_bert_embedding(token_humanish)
        
        sub_embeds = []
        for subtoken in token_humanish_base:
            sub_embed = get_bert_embedding(subtoken)
            sub_embeds.append(sub_embed)
        sub_embeds = torch.stack(sub_embeds)  # [num_subtokens, hidden_size] on CPU
        
        similarities = F.cosine_similarity(embed_full.unsqueeze(0), sub_embeds, dim=1)
        weights = F.softmax(similarities, dim=0)
        
        old_embeds_1 = torch.stack([model.get_input_embeddings().weight[old_id] for old_id in old_token_ids])
        weighted_sum_1 = (weights.unsqueeze(1) * old_embeds_1).sum(dim=0)        
        old_embeds_2 = torch.stack([model.get_output_embeddings().weight[old_id] for old_id in old_token_ids])
        weighted_sum_2 = (weights.unsqueeze(1) * old_embeds_2).sum(dim=0)
        
        # Assign weighted sum to both input and output embeddings
        new_input_embed[new_id] = weighted_sum_1
        new_output_embed[new_id] = weighted_sum_2
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.resize_token_embeddings(len(new_tokenizer))
    
    model.get_input_embeddings().weight.copy_(new_input_embed)
    model.get_output_embeddings().weight.copy_(new_output_embed)

bert_model.to("cpu")
torch.cuda.empty_cache()

model.to("cuda")

# Test the adapted model
test_input = "कृषि"  # Replace with a suitable test string
inputs = new_tokenizer([test_input], return_tensors="pt").to("cuda")
outputs = model.generate(inputs["input_ids"], max_new_tokens=128)
display_markdown(new_tokenizer.batch_decode(outputs)[0], raw=True)

print("Model adaptation complete!")
