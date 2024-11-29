import requests
import json
from utils import Response, show_reponse_metric

HOST = "http://localhost:5141"

def test_generate():
    # POST /api/generate
    url = f"{HOST}/api/generate"
    data = {
        "model": "llama3.2",
        "prompt": "why is the sky blue?",
        "stream": False
    }

    response = requests.post(url, json=data)
    pack_data = Response.from_dict(response.json())
    print(pack_data.response)
    show_reponse_metric(pack_data)

def test_generate_stream():
    # POST /api/generate
    url = f"{HOST}/api/generate"
    data = {
        "model": "llama3.2",
        "prompt": "why is the sky blue?",
    }
    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        for pack in response.iter_lines(512):
            if pack:
                try:
                    pack_data = Response.from_dict(json.loads(pack))
                    if not pack_data.done:
                        print(pack_data.response, end="", flush=True)
                    else:
                        print()
                        show_reponse_metric(pack_data)
                except Exception as e:
                    print("Error decoding JSON:", e)
    else:
        print("Error:", response.status_code, response.text)


def test_generate_with_img():
    # POST /api/generate
    from utils_image import Image2base64
    url = f"{HOST}/api/generate"
    data = {
        "model": "llava",
        "prompt": "What is in this picture?",
        "stream": False,
        "images": [Image2base64("background1.jpg")]
    }
    response = requests.post(url, json=data)
    pack_data = Response.from_dict(response.json())
    print(pack_data.response)
    show_reponse_metric(pack_data)

def test_embed():
    # POST /api/embed
    from utils import EmbeddingResponse
    url = f"{HOST}/api/embed"
    data = {
        # "model": "nomic-embed-text",
        "model": "all-minilm",
        "input": [
            "Why is the sky blue?", 
            "Why is the grass green?"
        ]
    }

    response = requests.post(url, json=data)
    pack_data = EmbeddingResponse.from_dict(response.json())
    print(pack_data.embeddings.shape)
    show_reponse_metric(pack_data)

if __name__ == "__main__":
    # test_generate()
    test_generate_stream()
    # test_generate_with_img()
    # test_embed()