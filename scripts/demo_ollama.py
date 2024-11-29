from ollama import Client
client = Client(host='http://localhost:11434')

def test_generate():
  response = client.generate(
    model = 'llama3.2', 
    prompt = "why is the sky blue?",
  )
  print(response)

def test_chat_stream():
  stream = client.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
  )

  for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

def test_embed():
  from utils import EmbeddingResponse, show_reponse_metric
  response = client.embed(
    # model='nomic-embed-text',
    model='all-minilm',
    input='This is a test',
  )
  pack_data = EmbeddingResponse.from_dict(response)
  print(pack_data.embeddings.shape)
  show_reponse_metric(pack_data)

def test_list_models():
  from utils import Model
  response = client.list()
  for model in response['models']:
    print(Model.from_dict(model).name)

if __name__ == '__main__':
  # test_generate()
  # test_chat_stream()
  # test_embed()
  test_list_models()

# python scripts\demo_ollama.py