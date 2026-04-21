# AI 100 - Homework 4

Interacting with an LLM API using code instead of a web interface.

## Model
Meta LLaMA 3.3 70B Instruct (open-source, via Hugging Face + Groq)

## Setup
```bash
pip install huggingface_hub
export HF_TOKEN="your_token_here"
python3 hw4_llm_api.py
```

## Experiments
1. Factual question about transformer neural networks
2. Same question with an expert system prompt
3. Creative prompt (poem)
4. Inspection of raw API response fields
