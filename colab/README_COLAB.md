# Running the AI Model on Google Colab

This guide explains how to run and fine-tune your AI Study Coach's LLM on Google Colab's GPU.

## Why Google Colab?

Your laptop has 4GB VRAM, which limits you to very small models. Google Colab provides a **T4 GPU (16GB VRAM)**, allowing you to run and fine-tune **DeepSeek-R1-Distill-Llama-8B**.

## Architecture

```
┌───────────────────────┐            ┌──────────────────────────────┐
│  Your Laptop          │            │  Google Colab (T4 GPU)       │
│                       │   HTTPS    │                              │
│  FastAPI Server ──────┼───────────►│  Ollama Server               │
│  (port 8000)          │   ngrok    │  (port 11434)                │
│                       │   tunnel   │                              │
│  Quiz App Frontend    │            │  Model: deepseek-r1:8b       │
│  (port 3000)          │            │  or: study-coach (fine-tuned) │
└───────────────────────┘            └──────────────────────────────┘
```

## What's in This Folder

| File | Description |
|------|-------------|
| `colab_ollama_server.py` | Inference notebook — run DeepSeek-R1 on Colab |
| `colab_finetune.py` | Fine-tuning notebook — train a custom study coach model |
| `training_data/study_coach_data.json` | Sample training data for fine-tuning |

## Prerequisites

1. **Google Account** with Google One (for better Colab GPU access)
2. **ngrok Account**: You already have this ✅
   - Auth token at: [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

---

## Option A: Run Pre-Trained DeepSeek-R1 (Quick Start)

Use `colab_ollama_server.py` to run the base model without fine-tuning.

### Steps

1. Go to [colab.research.google.com](https://colab.research.google.com/) → **New Notebook**
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Copy each cell from `colab_ollama_server.py` into the notebook
4. Run all cells in order

| Cell | Description | Time |
|------|-------------|------|
| 1 | Check GPU & Install Ollama | ~30 sec |
| 2 | Start Ollama Server | ~5 sec |
| 3 | Pull DeepSeek-R1 8B | ~3-5 min (first time) |
| 4 | Test the model | ~10 sec |
| 5 | Expose via ngrok | ~5 sec |
| 6 | Keep-alive loop | Runs forever |

After Cell 5, update your `.env`:

```env
COACH_OLLAMA_URL=https://xxxx-xxxx.ngrok-free.app
COACH_OLLAMA_MODEL=deepseek-r1:8b
```

---

## Option B: Fine-Tune a Custom Study Coach Model (Recommended for Thesis)

Use `colab_finetune.py` to create a model specifically trained for study coaching.

### What Fine-Tuning Does

- Takes the base DeepSeek-R1 model and trains it on **study coaching conversations**
- Uses **QLoRA** (4-bit quantization + LoRA) to fit in Colab's T4 GPU
- Only trains ~1-2% of the model's parameters (efficient!)
- Produces a custom model named `study-coach`

### Preparing Training Data

The sample data at `training_data/study_coach_data.json` contains 5 examples. For best results:

1. **Add more examples** (50-200 recommended)
2. Cover different scenarios:
   - Score analysis and feedback
   - Study plan creation
   - Progress tracking over time
   - Subject-specific advice
   - Motivational responses for struggling students
   - Spaced repetition recommendations
3. Keep the JSON format:

```json
{
    "messages": [
        {"role": "system", "content": "You are an AI Study Coach..."},
        {"role": "user", "content": "Student's question here"},
        {"role": "assistant", "content": "Coach's ideal response here"}
    ]
}
```

### Fine-Tuning Steps

1. **Prepare data**: Edit `training_data/study_coach_data.json` with your examples
2. Open a **new Colab notebook** with T4 GPU
3. **Upload** your training data file to Colab
4. Copy cells from `colab_finetune.py` into the notebook
5. Run cells in order:

| Cell | Description | Time |
|------|-------------|------|
| 1 | Install dependencies (Unsloth, etc.) | ~2-3 min |
| 2 | Load DeepSeek-R1 in 4-bit | ~2-3 min |
| 3 | Add LoRA adapters | ~5 sec |
| 4 | Prepare training data | ~5 sec |
| 5 | Fine-tune! | **~10-30 min** |
| 6 | Test fine-tuned model | ~30 sec |
| 7 | Export to GGUF format | ~5-10 min |
| 8 | Create Ollama model | ~1 min |
| 9 | Expose via ngrok | ~5 sec |
| 10 | Keep-alive loop | Runs forever |

After Cell 9, update your `.env`:

```env
COACH_OLLAMA_URL=https://xxxx-xxxx.ngrok-free.app
COACH_OLLAMA_MODEL=study-coach
```

---

## Troubleshooting

### "Ollama unreachable" on local server
- Check Colab notebook is running (Cell 6/10 shows ping logs)
- ngrok URL changes on restart — update `.env`
- Restart FastAPI server after updating `.env`

### Model is slow
- Run `!nvidia-smi` in Colab to check GPU usage
- DeepSeek-R1 8B takes ~3-8 seconds per response on T4

### Colab session disconnected
- Re-run all cells to get a new ngrok URL
- Google One sessions last ~12-24 hours
- The model download is cached within a session

### Fine-tuning out of memory
- Reduce `per_device_train_batch_size` to 1 in Cell 5
- Reduce `MAX_SEQ_LENGTH` to 1024 in Cell 2
- Reduce LoRA `r` from 16 to 8 in Cell 3

### GGUF export fails
- Make sure you have enough disk space (~10GB free)
- Try `q8_0` quantization for higher quality (larger file)

---

## For Your Thesis

### Without Fine-Tuning
> *"The AI Study Coach uses DeepSeek-R1-Distill-Llama-8B, a reasoning-capable LLM distilled from DeepSeek-R1. The model is self-hosted on a GPU-equipped cloud instance via the Ollama inference server, and accessed through a secure tunnel for low-latency communication with the application server."*

### With Fine-Tuning
> *"The AI Study Coach employs a fine-tuned version of DeepSeek-R1-Distill-Llama-8B, specialized for educational coaching through QLoRA (Quantized Low-Rank Adaptation). The model was trained on a curated dataset of study coaching conversations, enabling domain-specific responses including quiz performance analysis, personalized study plan generation, and spaced repetition scheduling. The fine-tuned model is quantized to GGUF format and served via Ollama on a cloud GPU instance."*

### Key Technical Terms for Thesis
- **QLoRA**: Quantized Low-Rank Adaptation — memory-efficient fine-tuning
- **LoRA rank (r=16)**: Controls adapter capacity
- **GGUF**: Quantized model format for efficient inference
- **Ollama**: Self-hosted LLM inference server
- **Spaced Repetition**: Learning technique the coach implements
