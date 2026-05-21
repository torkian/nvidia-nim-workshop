# Build Your First AI App with NVIDIA NIM

A 30-minute hands-on workshop. Students leave with a working Python AI app that calls an NVIDIA-hosted model and answers questions using their own context — the foundation of every RAG app.

**Audience:** university students, AI clubs, community devs. No GPU required. No CUDA. Just a browser.

---

## What you'll build

```
User question → Python app → NVIDIA NIM API → LLM response → App output
```

A tiny **campus assistant** that answers questions from a small knowledge base you give it.

---

## Prerequisites (5 minutes before the workshop)

1. **NVIDIA Developer account** — free: https://developer.nvidia.com/
2. **API key** from the API Catalog — https://build.nvidia.com/ → pick any model → "Get API Key"
3. **One of these**:
   - Easiest: a Google account (we'll use Colab — zero install)
   - Or: Python 3.10+ locally

That's it. No GPU, no Docker, no CUDA.

---

## Run it (two ways)

### Option A — Colab (recommended for the workshop)

Click: **[Open in Colab](#)** *(replace this link after you push to GitHub — see "Posting & sharing" below)*

In the first cell, paste your API key when prompted. Run all cells.

### Option B — Local Python

```bash
git clone https://github.com/torkian/01-nvidia-nim-workshop.git
cd 01-nvidia-nim-workshop
cp .env.example .env          # then edit .env and paste your NVIDIA_API_KEY
pip install -r requirements.txt
python app.py
```

---

## The 30-minute agenda

| Time | Segment | What happens |
|------|---------|--------------|
| 0–5  | Set the stage | What NIM is. Why APIs, not just chat UIs. The flow diagram. |
| 5–12 | First model call | Run the first cell. One API call. Change the system prompt. See it change. |
| 12–22| Campus assistant | Add a small knowledge base. Inject it as context. Answer questions about *your* data. |
| 22–27| Preview real RAG | Show that we just did manual retrieval. Next workshop: automate it with LangChain. |
| 27–30| Wrap & roadmap | The 5-workshop path. Where to go next. |

Full presenter script: see [`SLIDES.md`](SLIDES.md).
Student handout: see [`HANDOUT.md`](HANDOUT.md).

---

## Files in this repo

```
notebook.ipynb     — Colab-ready, the thing students run
app.py             — same logic as a local Python script
requirements.txt   — openai client + python-dotenv
.env.example       — template for your API key
SLIDES.md          — presenter notes / slide outline
HANDOUT.md         — 1-page student takeaway
```

---

## The workshop series

1. **Build Your First AI App with NVIDIA NIM** ← you are here
2. From AI App to RAG: Giving Your Model Knowledge (LangChain + NIM)
3. Guardrails & Evaluation
4. Run It Locally: NIM on Your Own GPU
5. From Chatbot to Agent

---

## Posting & sharing — where this should live

The goal: a student should be able to go from "saw the flyer" to "running code" in under 5 minutes.

**Tier 1 — the canonical home (do this first):**

- **GitHub public repo** — `github.com/torkian/01-nvidia-nim-workshop`
  - This is the link on the flyer, in the slides, and on your name tag.
  - Add a Colab badge to the README so students click *one button* to run.

- **Google Colab badge** — add this to the top of `README.md` once you know the repo URL:
  ```markdown
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/torkian/01-nvidia-nim-workshop/blob/main/notebook.ipynb)
  ```
  Colab pulls the notebook straight from GitHub. No download. No install.

**Tier 2 — discovery (post once the repo is live):**

- **NVIDIA Developer Forums** — `forums.developer.nvidia.com` → "Generative AI" category. Post: "Free 30-min university workshop: Build your first AI app with NVIDIA NIM (open-source materials)." Link the repo.
- **LinkedIn post from your account** — short, lead with the outcome ("students leave with a working AI app in 30 minutes"), link the repo, tag NVIDIA. This is the NVIDIA Champion signal.
- **University AI/CS club Discord or Slack** — pin the repo link.
- **Dev.to article** — write up the workshop as a tutorial. Cross-link to the repo. Dev.to indexes well for "NVIDIA NIM tutorial" searches.

**Tier 3 — durable distribution:**

- **YouTube** — record the 30-min run-through once. Put the repo link in the description. This becomes the asynchronous version for students who missed it.
- **A short URL** — register something like `bit.ly/nim-workshop-1` so you can put it on a slide and people can type it on a phone.

**What NOT to do:**
- Don't post a `.zip` or PDF — they go stale and there's no Colab button.
- Don't put the API key anywhere in the repo, ever. `.env` is gitignored on purpose.

---

## License

MIT — fork it, run it at your school, change the campus_info to your campus.
