# Presenter Script — Build Your First AI App with NVIDIA NIM

30 minutes. Keep it moving. Every minute of demo > every minute of theory.

---

## Slide 1 — Title (0:00)
**Build Your First AI App with NVIDIA NIM**
Subtitle: *From prompt to production-style inference in 30 minutes.*
You. Your name. NVIDIA Champion.

**Say:** "By the end of this session, you will not have *used* an AI tool. You will have *built* one."

---

## Slide 2 — Why this matters (0:30)
Three bullets:
- ChatGPT taught the world to *talk* to AI.
- Developers connect *code* to AI.
- Today: you become a developer.

**Say:** "Everything you'll see — every chatbot at every company you'll interview at — boils down to what we're about to do."

---

## Slide 3 — The flow (1:30)
Show this diagram, large:

```
User question → Python app → NVIDIA NIM API → LLM response → App output
```

**Say:** "An AI app is just code that asks an inference endpoint a question and does something with the answer. That's it. The 'AI' part is one HTTP call."

---

## Slide 4 — What is NIM? (2:30)
- **NIM = NVIDIA Inference Microservices.**
- Production-grade model endpoints, packaged by NVIDIA.
- Hosted in NVIDIA's API Catalog (we use this today) **or** run on your own GPU.
- Same code, same API, either way.

**Say:** "You write the app once. Where the model runs is a config change."

---

## Slide 5 — Open the link (5:00) — DEMO STARTS
Put the Colab link on screen. Have students click it. Wait until everyone has it open.

**Say while they load:** "If you can read this slide, you have everything you need. No installs. No GPU. No CUDA."

---

## Step 1 — First API call (5–12)

Walk through Cell 1 (auth) and Cell 2 (the `ask` function).

**Talk through it line by line:**
- `base_url='https://integrate.api.nvidia.com/v1'` → "this is where NVIDIA hosts the model"
- `model='meta/llama-3.1-8b-instruct'` → "any model on build.nvidia.com works here"
- `messages=[system, user]` → "this is the entire API"

Run it. Show the output.

**Ask the room:** "Now change the system prompt to a pirate. Anyone get a working pirate?"

This is where the room wakes up.

---

## Step 2 — System prompt is the steering wheel (12–17)

Run Cell 3. Same question, "sarcastic professor" persona.

**Say:** "Same model. Same question. The system prompt changed everything. This is the single most important lever in AI engineering."

---

## Step 3 — Campus assistant (17–22)

Run Cell 4. Show the three Q&A:
1. ✅ AI Club meeting — answered from data
2. ✅ Saturday lab hours — *correctly* inferred from data
3. ❌ Wifi password — *refused*

**Say:** "Look at answer 3. The model said 'I don't know.' That refusal — that's what separates a chatbot from an *app you can ship*."

**Then:** "We just did manual RAG. We retrieved (by hand) and we augmented (by hand). In Workshop 2, we automate the retrieval part."

---

## Slide 6 — The series (22–27)

```
1. ✅ First AI App with NIM     ← today
2. RAG with LangChain + NIM     ← next
3. Guardrails & Evaluation
4. Run NIM on your own GPU
5. From Chatbot to Agent
```

**Say:** "Each workshop adds *one* capability. Show up to all five and you'll have built an agent."

---

## Slide 7 — Take it home (27–29)

- Repo: `github.com/torkian/nvidia-nim-workshop`
- Fork it. Replace `campus_info` with *your* data. Show me on Discord.
- Join the NVIDIA Developer Program (free).
- Star the repo if it helped.

---

## Slide 8 — Q&A (29–30)

Have one backup question ready in case the room is quiet:
*"What model should I try next?"* → walk to build.nvidia.com, pick a reasoning model, show the difference.

---

## Things to remember while presenting

- **Don't read the code aloud.** Point and explain.
- **Don't apologize for skipping things.** "We're using llama-3.1-8b because it's fast. Try others on your own" — and move on.
- **If a student's key doesn't work** — have a backup key ready (revoke after). Don't let one student stall the room.
- **When something errors live** — say "good, this is real software" and move on. Don't debug live for more than 30 seconds.
