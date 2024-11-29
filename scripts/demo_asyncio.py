import asyncio
from ollama import AsyncClient

HOST='http://localhost:5141'

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient(HOST).chat(
      model='llama3.1', 
      messages=[message], 
      stream=True
    ):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())