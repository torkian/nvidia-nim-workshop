# Posting Plan — Share This Tutorial With the Community

The goal: turn this workshop into a **community resource** that helps people who will never sit in your room. Copy-paste templates below. Post in this order.

---

## Step 0 — Before you post anywhere

1. Push the repo to GitHub: `github.com/torkian/01-nvidia-nim-workshop`
2. Replace every `torkian` in the README, HANDOUT, and the Colab badge URL.
3. **Test the Colab badge yourself in an incognito window.** If it doesn't open the notebook in one click, fix it before posting.
4. Pick a clean repo description: *"30-minute hands-on workshop: build your first AI app with NVIDIA NIM. Free, open-source, Colab-ready."*
5. Add topics on GitHub: `nvidia`, `nim`, `llm`, `tutorial`, `workshop`, `rag`, `ai`, `python`, `colab`.

---

## Tier 1 — The high-signal posts (do these first, same day)

### 1. NVIDIA Developer Forums
**Where:** https://forums.developer.nvidia.com/ → **Generative AI** category.
**Why first:** official NVIDIA channel. Highest credibility. Counts toward the Champion path. NVIDIA staff sees it.

**Title:**
> Free 30-min workshop: Build your first AI app with NVIDIA NIM (open-source, Colab-ready)

**Body:**
```
Hi all — I put together a 30-minute hands-on workshop for university clubs and
new developers to build their first AI app using the NVIDIA API Catalog / NIM.

Everything runs in Colab — no GPU or CUDA setup required. Students leave with
a working Python app that calls an NVIDIA-hosted model and answers questions
from a knowledge base they provide (the foundation of every RAG app).

Repo (MIT): https://github.com/torkian/01-nvidia-nim-workshop
One-click Colab: <paste the Colab badge URL>

Included:
- Colab-ready notebook
- Local Python script
- 30-min presenter script
- 1-page student handout
- Series roadmap (RAG → guardrails → local NIM → agents)

Feedback welcome. If you run this at your school or meetup, ping me — I'll
link your version.

— B Torkian, NVIDIA Champion
```

### 2. LinkedIn (post, not article)
**Why:** the NVIDIA Champion signal. Tag NVIDIA. Short, lead with the outcome.

**Body:**
```
Just open-sourced a 30-minute workshop:
"Build Your First AI App with NVIDIA NIM."

For university AI clubs, CS classes, and community meetups.

Students leave with a working Python app that calls an NVIDIA-hosted model
and answers questions from their own data — the foundation of every RAG app.

No GPU. No CUDA. No installs. Just Colab.

Repo (MIT, fork it for your school):
github.com/torkian/01-nvidia-nim-workshop

If you teach, run an AI club, or organize meetups — take this, change the
campus data to yours, and run it. Let me know how it goes.

#NVIDIA #NIM #AI #Education
```

### 3. r/LocalLLaMA (Reddit)
**Why:** the most active LLM-builders community. They care about hands-on, runnable, free.

**Title:**
> Open-sourced a 30-min workshop teaching people to build their first AI app with NVIDIA NIM (Colab, no GPU needed)

**Body:**
```
Built this for a university workshop and figured the materials should be
public. It teaches a complete beginner to:

- call an NVIDIA-hosted LLM from Python via the OpenAI-compatible API
- understand what a system prompt actually does
- inject their own data and watch the model refuse out-of-scope questions

That refusal step is the punchline — it's what separates a toy from an app.

Everything runs in Colab. MIT license. Fork it for your group.

github.com/torkian/01-nvidia-nim-workshop

Feedback / PRs welcome. Workshop 2 (LangChain + NIM RAG) is next.
```

**Don't:** post the same wording on r/MachineLearning — that sub is strict about self-promo. Skip it or rewrite as a discussion question.

### 4. Hacker News — Show HN
**Title (exact format matters):**
> Show HN: A 30-minute workshop to build your first AI app with NVIDIA NIM

**Body (the "text" field):**
```
Materials I made for a university workshop, now open-sourced.

Goal: take someone who has only used ChatGPT and get them to a working Python
AI app in 30 minutes — calling an NVIDIA-hosted model and grounding it in
their own data. Everything runs in Colab so there's nothing to install.

Repo: https://github.com/torkian/01-nvidia-nim-workshop

The interesting part isn't the API call — it's the moment students see the
model refuse a question because the answer isn't in the data they provided.
That's the bridge from "chatbot" to "app you can ship."

Happy to take feedback on pacing and where it loses people.
```

**Time it well:** post 8–10am US Pacific on a Tuesday or Wednesday for best visibility.

---

## Tier 2 — Long-form (post within the week)

### 5. Dev.to article
**Why:** dev.to indexes extremely well for "NVIDIA NIM tutorial" type searches. Long shelf life.

**Title:** *Build Your First AI App with NVIDIA NIM in 30 Minutes*
**Tags:** `nvidia`, `ai`, `python`, `tutorial`
**Cover image:** the flow diagram from the README.

**Structure:**
1. Hook — "ChatGPT taught the world to *talk* to AI. Let's build one."
2. What NIM is, in 3 sentences.
3. The 5-line API call.
4. The system prompt experiment.
5. The campus assistant + the refusal moment.
6. What RAG is and where to go next.
7. Link the repo + Colab.

Cross-post the same article to **Hashnode** and **Medium** (set canonical URL to dev.to).

### 6. LinkedIn article (long-form, separate from the post above)
**Why:** different audience than the LinkedIn post. Educators, dept chairs, AI club faculty advisors read articles.
**Angle:** "How I'm teaching the next generation of AI developers — and the free materials any educator can use."

### 7. YouTube — async version
**Why:** the durable, searchable version. A year from now this is still pulling people to the repo.
**Format:** screen-record yourself running the Colab notebook end-to-end (15–20 min). Repo link in description.
**Title:** *Build Your First AI App with NVIDIA NIM (30-Min Workshop, Free Materials)*

---

## Tier 3 — Get into "awesome" lists (one-time effort, long tail)

Submit PRs adding your repo to:
- `awesome-nvidia` style lists on GitHub
- `awesome-llm` / `awesome-llm-apps`
- `awesome-rag` (Workshop 2 fits here even better)
- University AI-club resource lists (search GitHub for `awesome-ai-club`, `cs-resources`)

Each PR is 30 seconds. They each add a permanent inbound link.

---

## Tier 4 — Direct community help (this is where you actually help people)

The posts above are broadcast. *These* are where someone says "I'm stuck on X" and you say "I made this, try it."

- **NVIDIA Developer Discord / Slack** — respond when someone asks "how do I get started with NIM?"
- **r/learnmachinelearning** — only respond to specific "how do I start with NVIDIA AI" threads. Don't broadcast.
- **r/Python** beginner threads — same rule. Only when it's a direct fit.
- **University Discord servers** — your home turf. Pin it.
- **LangChain / LlamaIndex Discords** — for Workshop 2, this is where you go.
- **NVIDIA Developer Forums "I'm new" threads** — search every week, answer one or two.

Rule: **never paste the link without context.** Always answer the actual question first, then say "I made a 30-min workshop covering exactly this if you want a guided path." That's the difference between helping and spamming.

---

## Tracking what works

Add a simple UTM-style anchor to the repo URL per channel so you can see what drives traffic:
- `?utm=hn`, `?utm=reddit`, `?utm=linkedin`, `?utm=nvidia-forums`

GitHub's traffic page (Insights → Traffic) shows where stars and clones come from. Check it after 2 weeks. **Double down on the top channel for Workshop 2.**

---

## What "helping the community" looks like over 3 months

- Month 1: post the workshop. Run it once in person. Take notes on where students got stuck.
- Month 2: ship Workshop 2 (RAG). Now you have a *series*, which is more sharable than one post.
- Month 3: write up "What I learned teaching 200 students to use NVIDIA NIM." This is the post that goes far.

The repo is the seed. The series is the tree. The reflection post is the fruit.
