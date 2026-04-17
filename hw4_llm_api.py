"""
AI 100 - Homework #4
Interacting with an LLM API using code (Hugging Face Inference API via Groq)
Model used: meta-llama/Llama-3.3-70B-Instruct (open-source, free)
"""

import os
from huggingface_hub import InferenceClient

TOKEN = os.environ.get("HF_TOKEN", "your_token_here")

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    provider="groq",
    token=TOKEN,
)

def query_llm(prompt: str, system_instruction: str = "You are a helpful assistant."):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.7,
    )
    return response

def inspect_response(response) -> None:
    print("─" * 60)
    print(f"Model              : {response.model}")
    print(f"Finish reason      : {response.choices[0].finish_reason}")
    print(f"Prompt tokens      : {response.usage.prompt_tokens}")
    print(f"Completion tokens  : {response.usage.completion_tokens}")
    print(f"Total tokens       : {response.usage.total_tokens}")
    print("\n[Response text]\n")
    print(response.choices[0].message.content)
    print("─" * 60)

print("\n=== Experiment 1: Factual question ===")
prompt_1 = "Explain what a transformer neural network is in 3 sentences."
resp_1 = query_llm(prompt_1)
inspect_response(resp_1)

print("\n=== Experiment 2: Same question, expert system prompt ===")
system_expert = (
    "You are a machine learning professor at MIT. "
    "Use precise technical language and assume the student has a solid math background."
)
resp_2 = query_llm(prompt_1, system_instruction=system_expert)
inspect_response(resp_2)

print("\n=== Experiment 3: Creative prompt ===")
prompt_3 = "Write a short poem (4 lines) about the feeling of training a neural network for the first time."
resp_3 = query_llm(prompt_3)
inspect_response(resp_3)

print("\n=== Experiment 4: Raw response object fields ===")
print(f"Response ID        : {resp_3.id}")
print(f"Number of choices  : {len(resp_3.choices)}")
print(f"Choice[0] role     : {resp_3.choices[0].message.role}")

print("\n✅ All experiments complete.")
