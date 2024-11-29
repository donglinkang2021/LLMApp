import requests
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

OLLAMA_API = "http://localhost:5141/v1"

@dataclass
class Model:
    id: str
    created: int
    owned_by: str

def unpack(data:Dict):
    object_type = data['object']
    if object_type == 'list':
        return [unpack(item) for item in data['data']]
    elif object_type == 'model':
        return Model(data['id'], data['created'], data['owned_by'])
    elif object_type == 'embedding':
        return np.array(data[object_type])

def get_models() -> List[Model]:
    response = requests.get(f"{OLLAMA_API}/models")
    return unpack(response.json())

def get_models_llama3_2():
    response = requests.get(f"{OLLAMA_API}/models/llama3.2")
    return unpack(response.json())

def post_embeddings():
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "nomic-embed-text",
        "input": [
            "why is the sky blue?", 
            "why is the grass green?",
            "你好呀，你在干嘛呢？",
        ]
    }
    response = requests.post(
        f"{OLLAMA_API}/embeddings", 
        headers=headers, 
        json=data
    )
    return unpack(response.json())

def post_chat_completions():
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }
    response = requests.post(
        f"{OLLAMA_API}/chat/completions", 
        headers=headers, 
        json=data
    )
    print(response.json())

def test_get_models():
    models = get_models()
    for model in models:
        print(model)

def test_get_models_llama3_2():
    model = get_models_llama3_2()
    print(model)

def test_post_embeddings():
    embeddings = post_embeddings()
    for embedding in embeddings:
        print("shape:", embedding.shape)


if __name__ == "__main__":
    test_get_models()
    # test_get_models_llama3_2()
    # test_post_embeddings()
    # post_chat_completions()


