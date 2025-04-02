# TokenAdapt

TokenAdapt is a Tokenizer Transplantation Tool that allows users to seamlessly transplant tokenizers between language models while preserving semantic meaning. This tool is designed for users who want to adapt models for specific tasks or datasets without losing the integrity of the original embeddings.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/aloobun/tokenadapt.git
cd tokenadapt

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers tqdm
```

## Key Features

- 🔄 **Seamless Tokenizer Transplantation**: Easily transfer tokenizers between models.
- 🧠 **Intelligent Embedding Initialization**: Seemless experirence due to 0-shot intialization based on strong heuristics.
- 🔗 **Support for Tied and Untied Embeddings**: Flexibility in handling different embedding configurations.
- 🚀 **Efficient Caching System**: Reduces computation time by caching embeddings.
- 🎯 **Configurable Temperature**: Adjusts the expressiveness of the heuristics for embedding initialization.
- 🕒 **Fast and Efficient**: Optimized for speed and accuracy.

## Quick Start

To use the tool, run the following command with the required arguments:

```bash
python src/transplant.py \
    --model_path <original_model> \
    --new_tokenizer_path <new_tokenizer> \
    --new_model_name <output_model_name> \
    --hf_token <your_hf_token>
```

### Required Arguments

- `--model_path`: Path to the original model (Hugging Face model ID or local path).
- `--new_tokenizer_path`: Path to the new tokenizer (Hugging Face model ID or local path).
- `--new_model_name`: Name for the output model on Hugging Face Hub.
- `--hf_token`: Your Hugging Face authentication token.

### Optional Arguments

- `--temperature` (default: 0.3): Controls the expressiveness of the heuristic for embedding initialization.
  - Lower values (< 0.3) yield more expressive weights.
  - Higher values (> 0.3) produce more bland weights.
  - Range: 0.0 to 1.0.

- `--multiple_of` (default: 128): Size to pad the embedding matrix to, improving throughput when set to powers of 2.

- `--dtype` (default: "fp32"): Processing data type.
  - Options: "bf16", "fp16", "fp32".
  - Affects memory usage and computation speed.

- `--embedding_model_path` (default: "nomic-ai/nomic-embed-text-v2-moe"): Model used for embedding generation.

## Example Usage

### Basic Usage

```bash
python src/transplant.py \
    --model_path "Qwen/Qwen2.5-3B-Instruct" \
    --new_tokenizer_path "tinycompany/Adi-Bun-128K" \
    --new_model_name "tinycompany/Qwentify" \
    --hf_token "hf_..."
```

### Advanced Usage with Custom Settings

```bash
python src/transplant.py \
    --model_path "Qwen/Qwen2.5-3B-Instruct" \
    --new_tokenizer_path "tinycompany/Adi-Bun-128K" \
    --new_model_name "tinycompany/Qwentify" \
    --hf_token "hf_..." \
    --temperature 0.24 \
    --multiple_of 256 \
    --dtype "bf16" \
    --embedding_model_path "BAAI/bge-m3"
```

## License

This project is licensed under the Apache 2.0 License.



A detailed paper on the it will be available soon. You can find a summary on arXiv [here](https://arxiv.org/abs/XXXX.XXXX).

Copyright © 2025 IsNoobGrammer and aloobun
