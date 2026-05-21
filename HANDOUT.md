# Build Your First AI App with NVIDIA NIM — 1-page handout

## Run it again later
**Colab:** https://colab.research.google.com/github/torkian/01-nvidia-nim-workshop/blob/main/notebook.ipynb
**Repo:** `github.com/torkian/01-nvidia-nim-workshop`
**API key:** https://build.nvidia.com/ → any model → *Get API Key*

## The 10-line app you built

```python
from openai import OpenAI

client = OpenAI(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key='nvapi-...',  # from build.nvidia.com
)

resp = client.chat.completions.create(
    model='meta/llama-3.1-8b-instruct',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user',   'content': 'Hello!'},
    ],
)
print(resp.choices[0].message.content)
```

That's the whole thing. Everything else — RAG, agents, guardrails — is built on top of this.

## Three things to try this week
1. **Swap the data.** Replace `campus_info` with notes from one of your classes. Ask it questions before your next exam.
2. **Swap the model.** On build.nvidia.com, try a reasoning model and a code model. Same code, just change `MODEL = '...'`.
3. **Ship it.** Wrap the function in a Flask or FastAPI endpoint. You now have an AI API.

## Where this goes next
- **Workshop 2 — RAG with LangChain + NIM:** automate retrieval over many documents.
- **Workshop 3 — Guardrails & Evaluation:** make it safe to ship.
- **Workshop 4 — Run NIM locally on a GPU.**
- **Workshop 5 — From Chatbot to Agent.**

## Get plugged in
- NVIDIA Developer Program (free): https://developer.nvidia.com/
- NVIDIA Developer Forums → Generative AI category
- Star the repo, open issues, share what you built
