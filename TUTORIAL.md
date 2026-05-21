---
title: From First API Call to Working Agent: A Complete NVIDIA NIM Tutorial
published: false
description: Build a campus assistant in five stages: hosted NVIDIA NIM call, embedding RAG with query/passage inputs, guardrails, local NIM option, and a tool-calling agent.
tags: nvidia, ai, python, tutorial
cover_image: <upload a banner; use the flow diagram>
canonical_url:
---

# From First API Call to Working Agent: A Complete NVIDIA NIM Tutorial

Most students I've taught have used ChatGPT. Far fewer have called a model from code.

That is the gap this workshop is meant to close.

In the first 30 minutes, you'll call an NVIDIA-hosted language model from Python, pass it a small knowledge base, and make it answer only from that data. If you work through all five parts in one pass, budget more like 75-90 minutes. No GPU setup, no CUDA detour, no pretending a notebook is production. The goal is simple: write a normal Python program that talks to an LLM and gets useful text back.

I'm B Torkian, and I use this as a starter workshop for university and community groups. I've run a version of it with about 40 students. What usually surprises people is how ordinary the app feels. Most of it is normal software; one function call in the middle just happens to be weirdly powerful.

Everything runs in Google Colab because, for a room full of mixed laptops (I have made peace with this), boring setup wins.

This version keeps the original 30-minute workshop as Part 1, then keeps going. By the end, you will have moved from one API call to retrieval, guardrails, local NIM, and a small tool-using agent. It is still one tutorial, just with fewer excuses to stop right when it gets interesting.

---

## What you're building

```
User question → Python app → NVIDIA NIM API → LLM response → App output
```

We'll build a small campus assistant. It will:
- Call an NVIDIA-hosted Llama model
- Use the data you provide
- Retrieve relevant facts instead of pasting everything manually
- Refuse when the answer isn't there
- Call tiny Python tools when a plain answer is not enough

That refusal part matters. Demos can guess. Useful apps need to know when to say "I don't know."

---

## Part 1: Build your first AI app with NVIDIA NIM

This is the original starter workshop. It is deliberately small. If you are teaching this in a room, this is the part that gets everyone to the same working baseline before the clever stuff starts.

### What NVIDIA NIM is

NIM stands for NVIDIA Inference Microservices. For this workshop, treat it as hosted model inference from NVIDIA with a clean API in front.

There are two common ways to use it:

1. Hosted through NVIDIA's API Catalog at [build.nvidia.com](https://build.nvidia.com/). That's what we're using here; check the current catalog terms before you teach it, because credits and available models can change.
2. Self-hosted on your own GPU later, with the same API shape.

Whoever decided NVIDIA's API should mimic OpenAI's saved everyone a week of onboarding. You use the client most people have already seen, point it at a different endpoint, and move on.

### Prerequisites (5 minutes)

1. A free NVIDIA Developer account: [developer.nvidia.com](https://developer.nvidia.com/)
2. An API key from [build.nvidia.com](https://build.nvidia.com/) → pick any model → **Get API Key**. It starts with `nvapi-`.
3. A Google account for Colab.

The first time I taught this, I forgot to say the key starts with `nvapi-`, and half the room pasted the wrong thing (usually not their fault). Check that before you debug anything else.

### Step 1: Open Colab and install the client

NVIDIA's API Catalog is OpenAI-compatible, so we'll use the standard `openai` Python client and point it at NVIDIA's endpoint.

```python
%pip install -q openai

import os, getpass
from openai import OpenAI

os.environ['NVIDIA_API_KEY'] = getpass.getpass('Paste your NVIDIA API key: ')

client = OpenAI(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key=os.environ['NVIDIA_API_KEY'],
)

MODEL = 'meta/llama-3.1-8b-instruct'
```

Notice two things:

- `base_url` points at NVIDIA's hosted inference endpoint.
- `MODEL` is just a model name from the API Catalog. Swap it later if you want; the call shape does not change.

### Step 2: Make your first model call

```python
def ask(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_message},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content

print(ask(
    system_prompt='You are a helpful, concise assistant.',
    user_message='Explain GPU acceleration to a first-year CS student in 5 sentences.',
))
```

Run it.

That `ask()` function is the basic shape of a lot of AI apps: instructions in, user input in, model response out. Real systems add plumbing, but this is the core.

### Step 3: Use the system prompt to steer the model

Now keep the model and change its job description:

```python
print(ask(
    system_prompt='You are a sarcastic but accurate professor. Keep it under 5 sentences.',
    user_message='Explain GPU acceleration to a first-year CS student.',
))
```

The output changes because the system prompt changes the model's job. A little precision buys you a lot here; vague prompts make debugging miserable.

Treat prompts like tiny specs: include constraints, output shape, and what to do when a question goes off-track. Then test with slightly annoying questions, because users will absolutely ask those.

### Step 4: Build the campus assistant

An LLM doesn't know your campus schedule. It may still sound confident, which is exactly the problem.

So put the campus information directly into the prompt:

```python
campus_info = """
The AI Club meets every Thursday at 5 PM in the engineering building, room 204.
The GPU computing lab is open Monday to Friday from 10 AM to 6 PM.
Students can join the NVIDIA Developer Program for free.
The next workshop will cover Retrieval Augmented Generation (RAG).
Office hours for the AI/ML faculty are Tuesdays 2-4 PM.
"""

assistant_system_prompt = f"""You are a campus assistant. Answer ONLY using the
information in CAMPUS INFO below. If the answer is not in there, say
"I don't have that information — check with the AI Club."

CAMPUS INFO:
{campus_info}
"""

for question in [
    'When does the AI Club meet?',
    'Is the GPU lab open on Saturday?',
    'What is the wifi password?',
]:
    print(f'Q: {question}')
    print(f'A: {ask(assistant_system_prompt, question)}\n')
```

Run it and read the outputs before moving on. The AI Club answer should come straight from the text. For Saturday, the model often refuses with the fallback line instead of inferring that Saturday is closed. That is the behavior I want for this exercise: "Monday to Friday" gives a human enough to reason about Saturday, but the exact Saturday answer is not stated in the provided data.

The wifi question should also get the fallback line, because there is nothing in `campus_info` about passwords. If your model says "I don't have that information — check with the AI Club," do not treat that as a failure. It stayed inside the box we gave it, which is the whole point.

Last cohort, one student replaced the campus info with their D&D campaign notes and ended up with the most fun bug-hunting session of the day. The pattern works for silly data and useful data, which is why it sticks.

That refusal is the interesting bit. The model could guess, but we told it to stay inside the provided data. Do not oversell this part: the model can still make mistakes inside the text you gave it. It is just a concrete starting point for showing students where guardrails begin.

Try one more question of your own.

### Step 5: What you actually did

You just built manual RAG: you picked the context by hand, inserted it into the prompt, and asked the model to answer from that context. In a production-ish version, the hand-picked `campus_info` string becomes whatever your retrieval system finds.

In a real app, the context probably comes from PDFs, docs, tickets, lecture notes, or a wiki. You retrieve a few relevant chunks at query time, usually with embeddings and a vector database, then pass only those along.

The model call barely changes: `campus_info` becomes the output of retrieval. Most of the engineering work lives in that swap.

---

## Part 2: Give it real knowledge (manual RAG → embedding-based RAG)

Manual RAG works when the knowledge base is five lines long. It stops being cute when the campus handbook, club docs, and workshop notes start competing for the same prompt window, so the next move is retrieval: embed chunks, compare the user's question to those chunks, and pass only the closest few into the model. No vector database yet; a Python list and NumPy are enough to learn the idea.

```python
import numpy as np

EMBED_MODEL = 'nvidia/nv-embedqa-e5-v5'

knowledge_base = [
    {
        'title': 'AI Club meeting',
        'text': 'The AI Club meets every Thursday at 5 PM in the engineering building, room 204.',
    },
    {
        'title': 'GPU lab hours',
        'text': 'The GPU computing lab is open Monday to Friday from 10 AM to 6 PM.',
    },
    {
        'title': 'NVIDIA Developer Program',
        'text': 'Students can join the NVIDIA Developer Program for free.',
    },
    {
        'title': 'Next workshop',
        'text': 'The next workshop will cover Retrieval Augmented Generation (RAG).',
    },
    {
        'title': 'AI/ML office hours',
        'text': 'Office hours for the AI/ML faculty are Tuesdays 2-4 PM.',
    },
    {
        'title': 'Robotics lab',
        'text': 'The robotics lab requires safety training before students can use the soldering station.',
    },
    {
        'title': 'Tutoring',
        'text': 'Peer tutoring for introductory Python is available Wednesdays from 1 PM to 3 PM.',
    },
]

def embed_texts(texts: list[str], input_type: str = 'passage') -> list[np.ndarray]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={'input_type': input_type},
    )
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]

# Store knowledge chunks as passages. Embed the user's question as a query below.
# With the OpenAI Python client, that NVIDIA-specific option goes in extra_body.
embeddings = embed_texts([item['text'] for item in knowledge_base], input_type='passage')

for item, embedding in zip(knowledge_base, embeddings):
    item['embedding'] = embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)

def retrieve_context(question: str, k: int = 3) -> str:
    question_embedding = embed_texts([question], input_type='query')[0]

    scored = []
    for item in knowledge_base:
        score = cosine_similarity(question_embedding, item['embedding'])
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_items = [item for score, item in scored[:k]]

    return '\n'.join(f"- {item['text']}" for item in top_items)

def ask_with_retrieval(question: str) -> str:
    context = retrieve_context(question)

    system_prompt = f"""You are a campus assistant. Answer ONLY using the
context below. If the answer is not in the context, say
"I don't have that information — check with the AI Club."

CONTEXT:
{context}
"""

    return ask(system_prompt, question)

for question in [
    'Where does the AI Club meet?',
    'When can I get Python tutoring?',
    'What is the wifi password?',
]:
    print(f'Q: {question}')
    print(f'Context:\n{retrieve_context(question)}')
    print(f'A: {ask_with_retrieval(question)}\n')
```

What happened here is small but important. We asked NVIDIA's embedding model to turn every stored chunk into a `passage` vector and the user's question into a `query` vector. Cosine similarity gave us a cheap relevance score, and the top chunks became the context for the same `ask()` function from Part 1. A vector database would make storage, filtering, and scale nicer, but it would not change the basic idea.

---

## Part 3: Add guardrails so it doesn't lie

There are two basic guardrails I like to teach before bringing in a framework. First, tell the model what the boundary is: this assistant answers campus AI workshop questions, not everything a person can type into a box. Second, check the answer after the fact with a tiny verifier call.

This will not replace a real evaluation system. For a workshop, though, it gives students a second layer they can inspect and argue with.

```python
FALLBACK = 'I don\'t have that information — check with the AI Club.'

def answer_is_grounded(question: str, context: str, answer: str) -> bool:
    verdict = ask(
        system_prompt='You are a strict grounding verifier. Respond with only yes or no.',
        user_message=f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Is every factual claim in the ANSWER supported by the CONTEXT?""",
    )
    return verdict.strip().lower().startswith('yes')

def ask_guarded(question: str) -> str:
    context = retrieve_context(question)

    system_prompt = f"""You are a campus assistant for AI Club, GPU lab,
NVIDIA program, workshop, office hour, robotics lab, and tutoring questions.

Rules:
- Answer ONLY using the CONTEXT below.
- If the user asks about anything outside this campus assistant scope, say "{FALLBACK}"
- If the answer is not in the context, say "{FALLBACK}"
- Do not invent names, dates, rooms, links, passwords, or policies.

CONTEXT:
{context}
"""

    answer = ask(system_prompt, question)

    if not answer_is_grounded(question, context, answer):
        return FALLBACK

    return answer

for question in [
    'When does the AI Club meet?',
    'Can you write my breakup text?',
    'What is the wifi password?',
]:
    print(f'Q: {question}')
    print(f'A: {ask_guarded(question)}\n')
```

The first layer is the prompt. It tells the model what the assistant is allowed to talk about and gives it the exact fallback sentence. The second layer is the verifier, which calls the model again with a much narrower job: read the context, read the answer, and say whether the answer is grounded. I would not sell this as production safety by itself, but I do like it as a teaching move because it makes one thing obvious: sometimes you use an LLM to check another LLM call, and that is normal software too.

---

## Part 4: Run it on your own GPU with NIM

This part needs a GPU and Docker. Skip it if you don't have one.

Hosted NIM is the right start. Local NIM is what you try when you care about data locality, latency experiments, or learning what is actually inside the deployment box. The command shape from NVIDIA's deploy flow looks like this:

```bash
export NGC_API_KEY="paste-your-ngc-or-api-catalog-key-here"
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

docker run -it --rm --name llama-3.1-8b-instruct \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

Use the exact image and tag from the model's **Deploy** tab on [build.nvidia.com](https://build.nvidia.com/). The only Python change is the endpoint:

```python
client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='not-needed-for-local-dev',
)
```

That is the trick: same API shape, different place doing inference.

---

## Part 5: Turn it into an agent

"Agent" is an overloaded word, so here is the boring version: the model can choose a tool, your code runs the tool, and the result goes back to the model. In this tutorial, that is all I mean by agent.

We'll give it two toy tools. One gets the current time. One searches the campus info using the retriever from Part 2. The model decides whether it needs either one.

```python
import json
from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time(timezone: str = 'America/Los_Angeles') -> str:
    try:
        zone = ZoneInfo(timezone)
    except Exception:
        zone = ZoneInfo('UTC')

    return datetime.now(zone).strftime('%A, %B %d, %Y at %I:%M %p %Z')

def search_campus_info(query: str) -> str:
    return retrieve_context(query, k=3)

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Get the current time in an IANA time zone.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'timezone': {
                        'type': 'string',
                        'description': 'IANA time zone, such as America/Los_Angeles or UTC.',
                    },
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_campus_info',
            'description': 'Search the campus assistant knowledge base for relevant information.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The campus question or search phrase.',
                    },
                },
                'required': ['query'],
            },
        },
    },
]

available_tools = {
    'get_current_time': get_current_time,
    'search_campus_info': search_campus_info,
}

def ask_agent(question: str) -> str:
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a campus assistant. Use tools when they help. '
                'Answer from tool results. If the tools do not provide the answer, '
                f'say "{FALLBACK}"'
            ),
        },
        {'role': 'user', 'content': question},
    ]

    for _ in range(3):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice='auto',
            temperature=0.2,
            max_tokens=400,
        )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or '{}')

            if name not in available_tools:
                result = f'Tool {name} is not available.'
            else:
                result = available_tools[name](**arguments)

            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'name': name,
                'content': str(result),
            })

    return 'I hit the tool loop limit.'

for question in [
    'What time is it in Los Angeles?',
    'When does the AI Club meet?',
    'Can I get the wifi password?',
]:
    print(f'Q: {question}')
    print(f'A: {ask_agent(question)}\n')
```

The tools are intentionally tiny. The useful pattern is the loop: ask the model, inspect `tool_calls`, run the requested Python function, append a `tool` message, and ask the model again with the result included. That is enough to turn a chatbot-shaped program into an agent-shaped program. You still own the actual behavior, because the model only gets to call the tools you expose.

---

## Three things to try this week

1. **Use your own notes.** Replace `knowledge_base` with notes from one class. Ask it questions before your next exam.
2. **Try another model.** On [build.nvidia.com](https://build.nvidia.com/), change `MODEL = '...'` and compare the answers.
3. **Ship.** Put `ask_guarded()` or `ask_agent()` behind Flask or FastAPI. Keep it boring at first.

---

## Get the full workshop materials

Workshop materials are open source:

**Repo:** `github.com/torkian/01-nvidia-nim-workshop`
**One-click Colab:** [Open the notebook](<paste the Colab URL>)

Included: a notebook version, a local Python script, a 30-minute presenter script, and a 1-page student handout.

MIT licensed. Fork it. Change `campus_info` to your campus, your club, your project. Run the workshop wherever you are.

---

## What's next after this series

This is the series, all in one post: first API call, manual RAG, embedding-based retrieval, guardrails, local NIM, and a small function-calling agent.

Next I want to write about evaluation, deploying to Vercel, and multi-agent setups. Evaluation especially deserves its own piece, because the fastest way to ruin a good AI demo is to have no idea whether it got better or just got louder.

---

*I'm B Torkian (GitHub: torkian); I teach this workshop for students and community groups, and I still think the boring parts are where most of the value is.*
