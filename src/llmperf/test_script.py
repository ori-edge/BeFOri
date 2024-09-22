import requests

prompt = "Tell me a story about dogs."

response = requests.post(f"http://localhost:8000/?prompt={prompt}", stream=True)
response.raise_for_status()
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end="")