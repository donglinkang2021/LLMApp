from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:5141/v1/',
    api_key='ollama', # required but ignored
)

def test_chat():
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': 'Say this is a test',
            }
        ],
        model='llama3.2',
    )

    print(chat_completion.choices[0].message.content)

def test_chat_with_image():
    from utils_image import Image2base64
    chat_completion = client.chat.completions.create(
        model="llava",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", 
                 "image_url": "data:image/png;base64,"+Image2base64("background.jpg"),
                },
            ],
        }],
        max_tokens=300,
    )
    print(chat_completion.choices[0].message.content)

def test_generate():
    completion = client.completions.create(
        model="llama3.2",
        prompt="why is the sky blue?",
    )
    print(completion.choices[0].text)

def test_list_models():
    list_completion = client.models.list()
    for model in list_completion.data:
        print(model)

def test_retrieve_model():
    model = client.models.retrieve("llama3.2")
    print(model)

def test_embeddings():
    embeddings = client.embeddings.create(
        model="all-minilm",
        input=[
            "why is the sky blue?", 
            "why is the grass green?"
        ],
    )
    for emb in embeddings.data:
        print(len(emb.embedding))

if __name__ == '__main__':
    # test_chat()
    # test_chat_with_image()
    # test_generate()
    # test_list_models()
    # test_retrieve_model()
    test_embeddings()

# python demo_openai.py