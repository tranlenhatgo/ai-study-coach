# 🎓 Complete Beginner Guide: Fine-Tuning Your AI Study Coach

> **For:** Someone with zero AI/ML experience  
> **Time needed:** ~1 day total (spread across steps)  
> **Cost:** $0 (everything is free)  
> **Result:** Your own fine-tuned AI model running on your laptop

---

## Table of Contents

1. [What Are We Actually Doing?](#1-what-are-we-actually-doing)
2. [Prerequisites Checklist](#2-prerequisites-checklist)
3. [Step 1: Install Ollama on Your Laptop](#step-1-install-ollama-on-your-laptop)
4. [Step 2: Test the Base Model Locally](#step-2-test-the-base-model-locally)
5. [Step 3: Create Training Data](#step-3-create-training-data)
6. [Step 4: Set Up Google Colab](#step-4-set-up-google-colab)
7. [Step 5: Fine-Tune the Model](#step-5-fine-tune-the-model)
8. [Step 6: Download & Run Your Model](#step-6-download--run-your-model)
9. [Step 7: Connect to Your App](#step-7-connect-to-your-app)
10. [Troubleshooting](#troubleshooting)
11. [Understanding What You Did (For Your Thesis)](#understanding-what-you-did-for-your-thesis)

---

## 1. What Are We Actually Doing?

Think of it like this:

```
Gemma 4 E2B = A university graduate who knows a lot of general things
                but has never tutored students before

Fine-tuning  = Teaching that graduate HOW to be a good study coach
                by showing them 100-200 examples of great coaching

Your model   = A study coach specialist who knows exactly how to
                analyze quiz scores and give personalized advice
```

**What fine-tuning is NOT:**
- ❌ Training a model from scratch (that costs millions of dollars)
- ❌ Changing the model's core knowledge
- ❌ Complicated math you need to understand

**What fine-tuning IS:**
- ✅ Showing the model examples of "when a student says X, respond like Y"
- ✅ Running a script that someone else wrote (Unsloth library)
- ✅ Waiting ~15 minutes while it learns

---

## 2. Prerequisites Checklist

Before starting, make sure you have:

- [ ] **Google Account** — for Google Colab (free)
- [ ] **ngrok Account** — you already have this ✅
  - Get your auth token: https://dashboard.ngrok.com/get-started/your-authtoken
- [ ] **Your laptop** — Windows with RTX 3050 (4GB VRAM)
- [ ] **Internet connection** — for downloading models (~3 GB)

**You do NOT need:**
- ❌ Python ML knowledge
- ❌ GPU knowledge  
- ❌ Math knowledge
- ❌ Any paid subscriptions

---

## Step 1: Install Ollama on Your Laptop

Ollama is like a "model runner" — it loads AI models and lets your app talk to them.

### 1.1 Download Ollama

1. Go to **https://ollama.com/download**
2. Click **"Download for Windows"**
3. Run the installer (next, next, finish)

### 1.2 Verify It Works

Open **PowerShell** (or Terminal) and type:

```powershell
ollama --version
```

You should see something like `ollama version 0.x.x`. If you see an error, restart your computer and try again.

### 1.3 Pull the Base Model

This downloads Gemma 4 E2B (~3 GB) to your laptop:

```powershell
ollama pull gemma4:e2b
```

⏳ **Wait 2-5 minutes** — it's downloading the model.

### 1.4 Test It

```powershell
ollama run gemma4:e2b "What is 2+2?"
```

If it responds with an answer — congratulations, you have an AI model running on your laptop! 🎉

Type `/bye` to exit.

---

## Step 2: Test the Base Model Locally

Let's see how the **base model** (before fine-tuning) handles study coaching:

```powershell
ollama run gemma4:e2b "I scored 40% on my Math quiz. Help me improve."
```

📝 **Save this response somewhere!** You'll compare it later against your fine-tuned model to show improvement in your thesis.

Try a few more:
```powershell
ollama run gemma4:e2b "Create a weekly study plan for Biology and Chemistry"
ollama run gemma4:e2b "My scores: Math 80%, Science 50%, History 30%. What should I focus on?"
```

**Notice:** The responses are OK but generic. They don't feel like a specialized study coach. That's what fine-tuning will fix.

---

## Step 3: Create Training Data

This is the **most important step**. Better training data = better model.

### 3.1 What Training Data Looks Like

Each example is a conversation showing the model what a PERFECT response looks like:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are an AI Study Coach for an online quiz platform..."
        },
        {
            "role": "user", 
            "content": "I got 45% on Science. Help me."
        },
        {
            "role": "assistant",
            "content": "Let's work on improving! 🌍\n\n**Your Score: 45%**\n\n**Study Plan:**\n1. Review fundamentals..."
        }
    ]
}
```

### 3.2 You Already Have 5 Examples

Look at `colab/training_data/study_coach_data.json` — there are 5 examples. But 5 is too few. **You need 100-200 for good results.**

### 3.3 How to Create More Examples (The Easy Way)

Use ChatGPT, Gemini, or any AI chat to generate examples for you. Here's the prompt to use:

```
I'm building an AI Study Coach for an online quiz platform. 
Generate 10 training conversations in JSON format.

Each conversation should have:
- system: "You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans."
- user: A student asking about their quiz performance
- assistant: A helpful, friendly, structured response with emoji, markdown formatting, specific advice

Cover these scenarios:
1. Student scored low on a quiz
2. Student improved from last time  
3. Student wants a study plan
4. Student is strong in some subjects, weak in others
5. New student with no quiz history
6. Student failing repeatedly
7. Student asking about spaced repetition
8. Student wants to prepare for an exam
9. Student demotivated about scores
10. Student asking which quiz to take next

Format as a JSON array.
```

### 3.4 Organize Your Data

1. Generate 10-20 examples at a time using the prompt above
2. Copy the JSON into your file: `colab/training_data/study_coach_data.json`
3. **Review each one** — fix any that look weird or have wrong advice
4. Repeat until you have **100-200 examples**

### 3.5 Quality Checklist

For each example, check:
- [ ] Does the assistant response use **emoji** (🎉 📈 🔴)?
- [ ] Does it use **markdown formatting** (bold, headers, bullet points)?
- [ ] Does it give **specific, actionable advice** (not just "study more")?
- [ ] Does it **reference the student's actual scores**?
- [ ] Is it **encouraging** even when scores are bad?
- [ ] Is it **realistic** (not AI-sounding or robotic)?

### 3.6 Example of BAD vs GOOD Training Data

**❌ BAD** (too short, no formatting, generic):
```json
{"role": "assistant", "content": "You should study more and take the quiz again."}
```

**✅ GOOD** (structured, specific, encouraging):
```json
{"role": "assistant", "content": "Let's improve that Math score! 📊\n\n**Your Score: 45%** — Room for growth!\n\n**Study Plan:**\n1. **Review basics** — Focus on the topics you missed\n2. **Practice daily** — 20 min/day for 3 days\n3. **Re-take** — Target 65%+ on your next attempt\n\nYou got this! 💪"}
```

---

## Step 4: Set Up Google Colab

Google Colab gives you a free GPU in the cloud for fine-tuning.

### 4.1 Open Colab

1. Go to **https://colab.research.google.com/**
2. Sign in with your Google account
3. Click **"New Notebook"**

### 4.2 Enable GPU

1. Click **Runtime** (top menu)
2. Click **Change runtime type**
3. Select **T4 GPU** 
4. Click **Save**

### 4.3 Verify GPU

Paste this in the first cell and press **Shift+Enter** to run:

```python
!nvidia-smi
```

You should see something like `Tesla T4` with `15360MiB` VRAM.  
If you see "No GPU", go back to step 4.2.

### 4.4 Upload Your Training Data

1. In the left sidebar, click the **📁 folder icon**
2. Click the **⬆️ upload icon**
3. Upload your `study_coach_data.json` file

---

## Step 5: Fine-Tune the Model

Now the fun part — copy cells from `colab/colab_finetune.py` into your Colab notebook.

### How to Read colab_finetune.py

The file is divided into **10 cells**, separated by `# %%` markers. Each cell goes into a separate Colab cell.

```
# %%  ━━━━━━━━━━━━━━━
# CELL 1: Install Dependencies    ← This is Cell 1
# ━━━━━━━━━━━━━━━━━━━

(code here)

# %%  ━━━━━━━━━━━━━━━
# CELL 2: Load Base Model         ← This is Cell 2
# ━━━━━━━━━━━━━━━━━━━

(code here)
```

### Run Each Cell

Copy each cell into Colab and run with **Shift+Enter**. Wait for each to finish before running the next.

| Cell | What It Does | What You See | Time |
|------|-------------|-------------|------|
| **1** | Installs software | "✅ All dependencies installed!" | 2-3 min |
| **2** | Downloads the AI model | "✅ Model loaded!" | 2-3 min |
| **3** | Prepares model for training | "✅ LoRA adapters attached!" + trainable ratio ~1.5% | 5 sec |
| **4** | Loads your training data | "✅ Training data ready!" + number of examples | 5 sec |
| **5** | **THE ACTUAL TRAINING** | Progress bar, loss numbers going down | **10-20 min** |
| **6** | Tests the trained model | Model responds to a test question | 30 sec |
| **7** | Converts to Ollama format | "✅ GGUF file created! Size: ~3 GB" | 5-10 min |
| **8** | Creates Ollama model | "✅ Custom model 'study-coach' created!" | 1 min |
| **9** | Makes it accessible online | Shows a public URL | 5 sec |
| **10** | Keeps Colab alive | Pings every 30 seconds | ∞ |

### What to Look For During Training (Cell 5)

You'll see output like this:
```
Step 1/15: loss = 2.5432
Step 2/15: loss = 2.1876
Step 3/15: loss = 1.8234
...
Step 15/15: loss = 0.9123
```

**The `loss` number should go DOWN over time.** This means the model is learning.
- Loss > 2.0 = model is still learning
- Loss 1.0-2.0 = model is getting better  
- Loss < 1.0 = model has learned well
- Loss < 0.3 = might be overfitting (memorizing instead of learning) — this is bad

### After Cell 9: Copy the URL

You'll see something like:
```
🔗 Public URL: https://abc123.ngrok-free.app
🤖 Model: study-coach
```

**Keep this tab open!** Closing it stops the model.

---

## Step 6: Download & Run Your Model

After fine-tuning, you have two options to run the model:

### Option A: Run from Colab (Immediately)

Just update your `.env` file with the URL from Cell 9:

```env
COACH_OLLAMA_URL=https://abc123.ngrok-free.app
COACH_OLLAMA_MODEL=study-coach
```

This works instantly but Colab may disconnect after a few hours.

### Option B: Download and Run Locally (Permanent)

After Cell 7 finishes, the GGUF file is in Colab. Download it to your laptop:

1. In Colab's file browser (left sidebar 📁), navigate to `study_coach_gguf/`
2. Find the `.gguf` file (~3 GB)
3. Right-click → **Download**
4. Save it somewhere on your laptop (e.g., `C:\models\study-coach.gguf`)

Then create the Ollama model locally:

```powershell
# Create a Modelfile (paste this into a file called Modelfile)
# Or run this in PowerShell:

@"
FROM C:\models\study-coach.gguf

SYSTEM """You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans. Be friendly and use structured formatting."""

PARAMETER temperature 0.7
PARAMETER num_ctx 2048
"@ | Out-File -FilePath Modelfile -Encoding utf8

# Create the model in Ollama
ollama create study-coach -f Modelfile

# Test it!
ollama run study-coach "I scored 40% on Math. Help me improve."
```

Then update your `.env`:
```env
COACH_OLLAMA_URL=http://localhost:11434
COACH_OLLAMA_MODEL=study-coach
```

Now your fine-tuned model runs permanently on your laptop! No internet needed. 🎉

---

## Step 7: Connect to Your App

### 7.1 Update .env

```env
# For Colab-hosted model:
COACH_OLLAMA_URL=https://abc123.ngrok-free.app
COACH_OLLAMA_MODEL=study-coach

# OR for locally-running model:
COACH_OLLAMA_URL=http://localhost:11434
COACH_OLLAMA_MODEL=study-coach
```

### 7.2 Start Your Server

```powershell
cd c:\codespace\ai-study-coach
venv\Scripts\activate
python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### 7.3 Test It

Open your browser and go to:
```
http://localhost:8000/health
```

You should see the model status. Then try the chat:
```
http://localhost:8000/docs
```

Use the `/chat` endpoint to test a message.

---

## Troubleshooting

### "No GPU" in Colab
→ Runtime → Change runtime type → T4 GPU → Save

### Cell 1 fails with package errors
→ Try running Cell 1 again. Sometimes pip has temporary issues.

### Cell 2 takes forever / times out
→ The model is downloading (~3 GB). Make sure your Colab has internet and wait.

### Cell 5 shows "CUDA out of memory"
→ Change these values in Cell 5:
```python
per_device_train_batch_size=1,     # Change 2 → 1
gradient_accumulation_steps=8,     # Change 4 → 8
```
→ Or in Cell 2, change: `MAX_SEQ_LENGTH = 1024`  (was 2048)

### Training loss doesn't go down
→ Your training data might be inconsistent. Check that all examples follow the same format and style.

### Cell 7 (GGUF export) fails
→ Run `!df -h` in Colab to check disk space. You need ~10 GB free.
→ Try: restart the runtime and run from Cell 2 again (Cell 1 stays installed).

### ngrok gives "too many connections"
→ Free ngrok only allows 1 tunnel. Close any other ngrok tunnels first.
→ Or restart the Colab runtime and run from Cell 8.

### Model responses are garbage after fine-tuning
→ You probably need more training data. 5 examples ≠ enough. Aim for 100-200.
→ Check training loss — if it went below 0.1, you overfitted. Reduce epochs from 3 to 1.

### "Ollama unreachable" from your FastAPI server
→ Check Colab is still running (Cell 10 should show ping logs)
→ ngrok URL changes every restart — update `.env`
→ Restart your FastAPI server: `python -m uvicorn server.main:app --reload`

---

## Understanding What You Did (For Your Thesis)

Here's how to explain each step in academic terms:

### The Base Model
> *"We selected Google Gemma 4 E2B as our base model — a compact, edge-optimized language model with 2.3 billion effective parameters (5.1B total), released in March 2026. This model was chosen for its native function-calling support (enabling agentic behavior), small memory footprint (~3 GB in Q4 quantization), and Apache 2.0 license."*

### The Fine-Tuning Method
> *"We applied QLoRA (Quantized Low-Rank Adaptation) to fine-tune the base model. QLoRA combines 4-bit quantization with LoRA adapters, allowing fine-tuning of large models on consumer hardware. Only 1.5% of the model's parameters were trained (the LoRA adapters), while the remaining 98.5% were frozen — this prevents catastrophic forgetting while enabling domain specialization."*

### The Training Data
> *"A curated dataset of N study coaching conversations was created, covering scenarios including performance analysis, study plan generation, motivational support, and multi-subject prioritization. Each example follows a system-user-assistant format with structured markdown responses."*

### The Result
> *"The fine-tuned model demonstrates improved domain-specific response quality compared to the base model, with specialized formatting, quiz score analysis, and actionable study recommendations — while maintaining the base model's general language capabilities and native function-calling support for agentic interactions."*

### Key Numbers to Include

| Metric | Value |
|--------|-------|
| Base model | Gemma 4 E2B (2.3B effective params) |
| Fine-tuning method | QLoRA (4-bit + LoRA r=16) |
| Trainable parameters | ~1.5% of total |
| Training examples | N (your number) |
| Training time | ~15 min on T4 GPU |
| Training epochs | 3 |
| Learning rate | 2e-4 |
| Quantization (inference) | Q4_K_M (~3 GB) |
| Inference VRAM | ~3 GB (fits RTX 3050) |
| Training loss (final) | Your number from Cell 5 |

---

## Quick Reference: The Whole Process

```
Day 1 (2-3 hours):
  ├── Install Ollama on laptop              (10 min)
  ├── Test base model                       (10 min)
  └── Create training data (100+ examples)  (2+ hours)

Day 2 (1-2 hours):
  ├── Open Google Colab                     (5 min)
  ├── Upload training data                  (2 min)
  ├── Run Cells 1-5 (fine-tune)            (30 min)
  ├── Run Cells 6-7 (test + export)        (15 min)
  ├── Download GGUF to laptop              (10 min)
  ├── Create local Ollama model            (5 min)
  ├── Update .env and test                 (10 min)
  └── 🎉 Done!
```

**Total hands-on time: ~4-5 hours**  
**Most of that is creating training data, not technical work.**
