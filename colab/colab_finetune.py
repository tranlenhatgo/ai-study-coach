# ============================================================================
# AI Study Coach — Fine-Tuning Gemma 4 E2B on Google Colab
# ============================================================================
#
# This script fine-tunes Google's Gemma 4 E2B using QLoRA (4-bit
# quantization + LoRA adapters) to specialize it as an AI Study Coach.
#
# Why Gemma 4 E2B?
#   - 2.3B effective params — runs on RTX 3050 (4GB VRAM) locally!
#   - Native function calling — supports agentic tool use
#   - Multimodal (text + image + audio)
#   - 128K context window
#   - Apache 2.0 license
#   - Released March 2026 — Google's latest
#
# After fine-tuning, the model is exported to GGUF format and loaded into
# Ollama as a custom model for serving.
#
# REQUIREMENTS:
#   - Google Colab with T4 GPU (16GB VRAM) — minimum
#   - ~10 GB free disk space
#   - Training data in colab/training_data/study_coach_data.json
#
# HOW TO USE:
#   1. Prepare your training data (see training_data/study_coach_data.json)
#   2. Upload training data to Colab
#   3. Run cells in order
#   4. After fine-tuning, run colab_ollama_server.py with the custom model
#
# ============================================================================


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1: Install Dependencies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# We use Unsloth for 2x faster fine-tuning with 60% less VRAM.
# This makes it possible to fine-tune on a free T4 GPU.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess
import torch

# Check GPU
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
print()

# Install unsloth (optimized fine-tuning library)
print("📦 Installing Unsloth + dependencies (this takes ~2-3 minutes)...")
subprocess.run([
    "pip", "install", "-q",
    "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git",
], check=True)
subprocess.run([
    "pip", "install", "-q",
    "xformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets",
], check=True)

print("✅ All dependencies installed!")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2: Load Base Model with 4-bit Quantization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# We load Gemma 4 E2B in 4-bit (QLoRA) to fit in T4's 16GB.
# Unsloth handles the quantization and optimization automatically.
#
# Gemma 4 E2B specs:
#   - 2.3B effective parameters (5.1B total with embeddings)
#   - Native tool calling / function calling support
#   - Only ~3 GB VRAM for inference (Q4) — fits on RTX 3050!
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/gemma-4-e2b"  # Unsloth-optimized Gemma 4 E2B
MAX_SEQ_LENGTH = 2048  # Context window for training
DTYPE = None           # Auto-detect (float16 for T4)
LOAD_IN_4BIT = True    # QLoRA — 4-bit quantization

print(f"📥 Loading {MODEL_NAME} in 4-bit mode...")
print("   This may take 2-3 minutes on first run (downloading model)...\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print(f"\n✅ Model loaded! Memory usage: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3: Add LoRA Adapters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# LoRA (Low-Rank Adaptation) only trains a small set of adapter weights
# instead of the full model. This makes fine-tuning feasible on consumer GPUs.
#
# Key parameters:
#   r = 16          → Rank of LoRA matrices (higher = more capacity, more VRAM)
#   lora_alpha = 16 → Scaling factor
#   target_modules  → Which layers to apply LoRA to
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("🔧 Adding LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,       # Optimized — Unsloth supports 0 dropout
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=42,
)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters:     {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Trainable ratio:      {trainable_params / total_params:.2%}")
print(f"\n✅ LoRA adapters attached!")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4: Prepare Training Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Upload your training data file: colab/training_data/study_coach_data.json
# Or use the built-in sample data to get started.
#
# The training data format is a list of conversations, where each conversation
# has a system prompt, user message, and assistant response.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
import os
from datasets import Dataset

# ── Use Gemma 4's native chat template ──
# Gemma 4 uses the standard gemma chat template which Unsloth handles
# automatically. We just need to ensure the tokenizer has it set.

# ── Load training data ──
DATA_FILE = "study_coach_data.json"

if os.path.exists(DATA_FILE):
    print(f"📂 Loading training data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"   Loaded {len(raw_data)} training examples")
else:
    print("📂 No training data file found. Using built-in sample data...")
    print("   💡 Upload 'study_coach_data.json' for better results!\n")

    # Sample training data — replace with your own for better results
    raw_data = [
        {
            "messages": [
                {"role": "system", "content": "You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans."},
                {"role": "user", "content": "I got 45% on my Science quiz about the Solar System. Help me improve."},
                {"role": "assistant", "content": "Let's work on improving your Solar System knowledge! 🌍\n\n**Your Score: 45%** — There's room for growth, and I'm here to help.\n\n**Study Plan:**\n1. **Review fundamentals** — Start with the order of planets and their key characteristics\n2. **Focus on weak areas** — Based on a 45% score, you likely need to review:\n   - Planet classifications (terrestrial vs gas giants)\n   - Orbital mechanics basics\n   - Key facts about each planet\n3. **Practice strategy** — Take the quiz again after studying each section\n\n**Recommended approach:**\n- Study for 20 minutes daily for the next 3 days\n- Re-take the quiz on day 4\n- Target: 70%+ on your next attempt\n\nWould you like me to break down any specific topic?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans."},
                {"role": "user", "content": "My quiz history shows: Math 85%, Science 60%, History 40%, English 90%. What should I focus on?"},
                {"role": "assistant", "content": "Great job on English (90%) and Math (85%)! 🎉 Let's focus on your weaker areas.\n\n**Performance Overview:**\n- 🟢 English: 90% — Excellent!\n- 🟢 Math: 85% — Strong\n- 🟡 Science: 60% — Needs improvement\n- 🔴 History: 40% — Priority focus\n\n**Recommended Study Priority:**\n\n### 1. History (40%) — Urgent\n- This is your weakest category and needs immediate attention\n- Spend 30 min/day reviewing History topics\n- Try active recall: read a section, close the book, write down what you remember\n- Re-take History quizzes every 3 days\n\n### 2. Science (60%) — Moderate\n- You have a foundation but need to fill gaps\n- Spend 20 min/day on Science after History study\n- Focus on the specific topics where you lost points\n\n### 3. Math & English — Maintain\n- Quick 10-min review sessions to keep these strong\n- Re-take quizzes weekly to maintain your scores\n\n**Weekly Schedule:**\n- Mon/Wed/Fri: History (30 min) + Science (20 min)\n- Tue/Thu: Science (30 min) + Math review (10 min)\n- Weekend: Mixed review + re-take weakest quiz\n\nShall I create a more detailed plan for History?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans."},
                {"role": "user", "content": "I keep failing the Geography quiz. I've taken it 3 times: 30%, 35%, 40%. Am I improving?"},
                {"role": "assistant", "content": "Yes, you ARE improving! 📈 Let's look at the trend:\n\n**Your Progress:**\n- Attempt 1: 30%\n- Attempt 2: 35% (+5%)\n- Attempt 3: 40% (+5%)\n\n**The good news:** You're consistently gaining 5% each attempt. That's real progress!\n\n**The challenge:** At this rate, you'd need several more attempts to reach a passing score. Let's accelerate your learning.\n\n**Accelerated Study Plan:**\n\n1. **Identify your mistake patterns** — Are you missing the same types of questions?\n   - Map-based questions?\n   - Capital cities?\n   - Physical geography (rivers, mountains)?\n\n2. **Targeted review** — Instead of re-reading everything:\n   - Review ONLY the questions you got wrong\n   - Write down the correct answers and WHY they're correct\n   - Quiz yourself on just those topics before retaking\n\n3. **Spaced repetition** — Don't retake immediately:\n   - Study today, review tomorrow, retake in 2 days\n   - This helps move information to long-term memory\n\n**Target:** 55%+ on your next attempt (a 15% jump instead of 5%)\n\nYou're showing persistence, and that's the most important skill! 💪 Keep going."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans."},
                {"role": "user", "content": "Create a weekly study plan for me. I'm weak in Biology and Chemistry but good at Physics."},
                {"role": "assistant", "content": "Here's your personalized weekly study plan! 📅\n\n**Focus areas:** Biology 🔴 & Chemistry 🟡 (priority) | Physics 🟢 (maintain)\n\n---\n\n### 📅 Monday — Biology Deep Dive\n- 🕐 30 min: Study weakest Biology topic\n- 🕐 15 min: Practice quiz (Biology section only)\n- 🕐 10 min: Write summary notes of what you learned\n\n### 📅 Tuesday — Chemistry Focus\n- 🕐 30 min: Study weakest Chemistry topic\n- 🕐 15 min: Practice problems/flashcards\n- 🕐 10 min: Physics quick review (keep skills sharp)\n\n### 📅 Wednesday — Biology + Review\n- 🕐 25 min: New Biology topic\n- 🕐 15 min: Review Monday's Biology notes (spaced repetition)\n- 🕐 10 min: Chemistry flashcard review\n\n### 📅 Thursday — Chemistry + Practice\n- 🕐 25 min: New Chemistry topic\n- 🕐 15 min: Review Tuesday's Chemistry notes\n- 🕐 15 min: Take a mixed Science practice quiz\n\n### 📅 Friday — Weak Spots\n- 🕐 20 min: Re-study any topics you struggled with this week\n- 🕐 20 min: Take Biology quiz (full)\n- 🕐 15 min: Take Chemistry quiz (full)\n\n### 📅 Weekend — Light Review\n- 🕐 20 min Saturday: Review all notes from the week\n- 🕐 20 min Sunday: Take Physics quiz to maintain your score\n\n---\n\n**Total study time:** ~4.5 hours/week\n**Expected improvement:** 10-15% per subject per week\n\nWant me to adjust the schedule based on your available time?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans."},
                {"role": "user", "content": "Hello! I just signed up. I haven't taken any quizzes yet."},
                {"role": "assistant", "content": "Welcome to the quiz platform! 👋 I'm your AI Study Coach.\n\nSince you haven't taken any quizzes yet, here's how to get started:\n\n**Step 1: Take Your First Quiz**\n- Browse the available quiz categories\n- Start with a subject you feel comfortable with — this gives us a baseline\n- Don't stress about the score! The first quiz helps me understand where you are\n\n**Step 2: Explore Other Subjects**\n- Take quizzes in 2-3 different categories\n- This helps me identify your strengths and weaknesses\n\n**Step 3: Come Back to Me**\n- After you've taken a few quizzes, I'll be able to:\n  - 📊 Analyze your performance across subjects\n  - 📋 Create a personalized study plan\n  - 🎯 Recommend which quizzes to focus on\n  - 📈 Track your improvement over time\n\n**Pro tip:** Answer honestly — guessing randomly won't help me help you!\n\nGood luck on your first quiz! 🍀 Come back anytime for study advice."}
            ]
        },
    ]

print(f"   Total training examples: {len(raw_data)}")

# ── Format conversations for training ──
def format_conversations(examples):
    """Apply chat template to each conversation."""
    texts = []
    for example in examples:
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts

formatted_texts = format_conversations(raw_data)
dataset = Dataset.from_dict({"text": formatted_texts})

print(f"   Formatted dataset: {len(dataset)} examples")
print(f"\n📄 Sample training text (first 300 chars):")
print("-" * 50)
print(dataset[0]["text"][:300])
print("-" * 50)
print("\n✅ Training data ready!")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5: Fine-Tune the Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Training configuration:
#   - Epochs: 3 (increase for more training data)
#   - Batch size: 2 (limited by T4 VRAM)
#   - Learning rate: 2e-4 (standard for LoRA)
#   - Warmup steps: 5
#
# Expected time: ~10-20 minutes depending on dataset size.
# Gemma 4 E2B is smaller than 8B models, so training is faster!
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from trl import SFTTrainer
from transformers import TrainingArguments

OUTPUT_DIR = "./study_coach_finetuned"

print("🏋️ Starting fine-tuning...")
print(f"   Dataset size: {len(dataset)} examples")
print(f"   Output dir:   {OUTPUT_DIR}")
print(f"   This may take 10-20 minutes on T4 GPU...\n")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Can set to True for short sequences
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",  # Disable wandb/tensorboard
    ),
)

# Train!
train_result = trainer.train()

# Print results
print("\n" + "=" * 60)
print("🎉 FINE-TUNING COMPLETE!")
print("=" * 60)
print(f"   Training loss:  {train_result.training_loss:.4f}")
print(f"   Training time:  {train_result.metrics['train_runtime']:.0f} seconds")
print(f"   Samples/second: {train_result.metrics['train_samples_per_second']:.1f}")
print(f"   GPU memory:     {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB peak")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6: Test the Fine-Tuned Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from unsloth import FastLanguageModel

# Switch to inference mode
FastLanguageModel.for_inference(model)

test_messages = [
    {"role": "system", "content": "You are an AI Study Coach for an online quiz platform. Analyze quiz performance and create personalized study plans."},
    {"role": "user", "content": "I scored 55% on Math and 80% on English. Create a study plan for this week."},
]

inputs = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

print("🧪 Testing fine-tuned model...\n")
print("User: I scored 55% on Math and 80% on English. Create a study plan for this week.\n")
print("🤖 Fine-tuned Gemma 4 E2B response:")
print("-" * 50)

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)
print("-" * 50)
print("\n✅ Fine-tuned model is working!")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7: Export to GGUF Format (for Ollama)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Convert the fine-tuned model to GGUF format so Ollama can load it.
# We use Q4_K_M quantization — the model will be ~3 GB, perfect for
# your RTX 3050 (4GB VRAM).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GGUF_OUTPUT = "./study_coach_gguf"
QUANT_METHOD = "q4_k_m"  # Good balance: quality vs size vs speed

print(f"📦 Exporting to GGUF format ({QUANT_METHOD})...")
print("   This may take 5-10 minutes...\n")

# Save in GGUF format using Unsloth's built-in converter
model.save_pretrained_gguf(
    GGUF_OUTPUT,
    tokenizer,
    quantization_method=QUANT_METHOD,
)

# Find the GGUF file
import glob
gguf_files = glob.glob(f"{GGUF_OUTPUT}/*.gguf")
if gguf_files:
    gguf_path = gguf_files[0]
    file_size_gb = os.path.getsize(gguf_path) / (1024**3)
    print(f"\n✅ GGUF file created: {gguf_path}")
    print(f"   Size: {file_size_gb:.1f} GB")
    print(f"   This will fit comfortably on your RTX 3050 (4GB VRAM)!")
else:
    print("⚠️ GGUF file not found. Check output above for errors.")
    gguf_path = None


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8: Create Custom Ollama Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Creates an Ollama model from the GGUF file so you can serve it just like
# any other Ollama model. The custom model is named "study-coach".
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess
import os

CUSTOM_MODEL_NAME = "study-coach"

if gguf_path is None:
    print("❌ No GGUF file found. Run Cell 7 first.")
    raise SystemExit("GGUF export required")

# Install Ollama binary (direct download — install.sh fails on Colab)
install_path = "/usr/local/bin/ollama"
if not os.path.exists(install_path):
    print("📦 Installing Ollama...")
    arch = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=True).stdout.strip()
    ollama_arch = "amd64" if "x86_64" in arch else "arm64"
    subprocess.run(
        ["curl", "-L", f"https://ollama.com/download/ollama-linux-{ollama_arch}", "-o", install_path],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(["chmod", "+x", install_path], check=True)
    print("✅ Ollama installed!")

# Make sure Ollama is running (start it if needed)
try:
    import urllib.request
    urllib.request.urlopen("http://localhost:11434", timeout=5)
except Exception:
    print("🚀 Starting Ollama server...")
    ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"},
    )
    import time
    time.sleep(5)

# Create Modelfile for Ollama
modelfile_content = f"""FROM {os.path.abspath(gguf_path)}

SYSTEM \"\"\"You are an AI Study Coach for an online quiz platform. You can both give advice AND take actions on the platform.

Your capabilities:
1. Analyze student quiz performance and identify weak areas
2. Create personalized study plans based on quiz history
3. Navigate students to specific pages (dashboard, quiz list, profile)
4. Start quizzes for students to practice
5. Generate questions on specific topics
6. Create practice quizzes targeting weak areas
7. Search for quizzes by category

Be friendly, supportive, and use structured formatting with bullet points.
When you take an action, briefly explain what you did and why.
Reference specific quiz results when giving advice.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
"""

modelfile_path = "/tmp/Modelfile.study-coach"
with open(modelfile_path, "w") as f:
    f.write(modelfile_content)

print(f"📦 Creating Ollama model '{CUSTOM_MODEL_NAME}' from fine-tuned GGUF...")
result = subprocess.run(
    ["ollama", "create", CUSTOM_MODEL_NAME, "-f", modelfile_path],
    capture_output=False,
)

if result.returncode == 0:
    print(f"\n✅ Custom model '{CUSTOM_MODEL_NAME}' created!")
    print(f"\n📋 Available models:")
    subprocess.run(["ollama", "list"])
    print(f"\n🎉 You can now use COACH_OLLAMA_MODEL={CUSTOM_MODEL_NAME} in your .env!")
else:
    print("❌ Failed to create Ollama model. Check the output above.")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9: Expose Fine-Tuned Model via ngrok
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Same as the inference notebook — expose via ngrok so your FastAPI server
# can connect to the fine-tuned model.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess

NGROK_AUTH_TOKEN = ""  # ← Paste your ngrok auth token here

if not NGROK_AUTH_TOKEN:
    print("❌ Please set your NGROK_AUTH_TOKEN above!")
    raise SystemExit("ngrok token required")

subprocess.run(["pip", "install", "pyngrok", "-q"], check=True)
from pyngrok import ngrok

ngrok.set_auth_token(NGROK_AUTH_TOKEN)
tunnel = ngrok.connect(11434)
public_url = tunnel.public_url

CUSTOM_MODEL_NAME = "study-coach"

print("=" * 60)
print("🎉 FINE-TUNED STUDY COACH IS NOW LIVE!")
print("=" * 60)
print()
print(f"   🔗 Public URL:  {public_url}")
print(f"   🤖 Model:       {CUSTOM_MODEL_NAME} (fine-tuned Gemma 4 E2B)")
print()
print("   Update your .env file:")
print()
print(f"   COACH_OLLAMA_URL={public_url}")
print(f"   COACH_OLLAMA_MODEL={CUSTOM_MODEL_NAME}")
print()
print("=" * 60)


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10: Keep Alive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import time
import urllib.request
from datetime import datetime

print("♻️  Keep-alive loop started.")
print(f"   Model: {CUSTOM_MODEL_NAME} (fine-tuned Gemma 4 E2B)")
print(f"   URL: {public_url}")
print("   Press Runtime → Interrupt execution to stop.\n")

ping_count = 0
while True:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=10)
        ping_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        if ping_count % 10 == 0:
            print(f"   [{timestamp}] ✅ Server healthy — {ping_count} pings")
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"   [{timestamp}] ⚠️ Health check failed: {e}")
    time.sleep(30)
