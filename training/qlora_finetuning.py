# =====================================================================
# QLoRA Fine-Tuning: English to Marathi Translation Script (OOM Fixed)
# =====================================================================

import os
import subprocess
import sys

# ---------------------------------------------------------------------
# Step 0: Install dependencies and set environment flags
# ---------------------------------------------------------------------
print("Checking and installing necessary libraries...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=5.12", "datasets", "peft", "accelerate", "bitsandbytes",
])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bitsandbytes as bnb
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Step 1: Configuration Variables (Optimized for T4 15GB VRAM)
# ---------------------------------------------------------------------
MODEL_ID = "bodhan-ai/indic-translate"
OUTPUT_DIR = "/kaggle/working/indic-translate-marathi-lora"

TARGET_LANGUAGE = "Marathi"
NUM_EPOCHS = 3
BATCH_SIZE = 2       # Reduced from 4 to save VRAM
GRAD_ACCUM = 2       # Increased to maintain effective batch size
LEARNING_RATE = 1e-4
MAX_LEN = 128        # Reduced from 256 to save activation memory
ADD_TOKEN_TYPE_IDS = False

# Log into Hugging Face (needed for gated/protected models)
login("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")


# ---------------------------------------------------------------------
# Step 2: Load and Preprocess Training Dataset (Samanantar 15k)
# ---------------------------------------------------------------------
print("Loading Samanantar train dataset...")
train_raw_dataset = load_dataset(
    "ai4bharat/samanantar",
    "mr",
    split="train"
)
train_df = train_raw_dataset.to_pandas()
print(f"Initial train dataset size: {len(train_df):,}")

# Clean Nulls & Empty Strings
train_df = train_df.dropna(subset=["src", "tgt"])
train_df["src"] = train_df["src"].astype(str).str.strip()
train_df["tgt"] = train_df["tgt"].astype(str).str.strip()
train_df = train_df[(train_df["src"] != "") & (train_df["tgt"] != "")]

# Remove Duplicates
train_df = train_df.drop_duplicates(subset=["src", "tgt"])
train_df = train_df.drop_duplicates(subset=["src"])

# Calculate Word Counts & Length Ratios
train_df["src_words"] = train_df["src"].str.split().str.len()
train_df["tgt_words"] = train_df["tgt"].str.split().str.len()
train_df["length_ratio"] = train_df["tgt_words"] / train_df["src_words"].clip(lower=1)

# Filter criteria
MIN_WORDS = 4
MAX_WORDS = 75
MIN_RATIO = 0.3
MAX_RATIO = 3.0

filtered_df = train_df[
    (train_df["src_words"] >= MIN_WORDS) & (train_df["src_words"] <= MAX_WORDS) &
    (train_df["tgt_words"] >= MIN_WORDS) & (train_df["tgt_words"] <= MAX_WORDS) &
    (train_df["length_ratio"] >= MIN_RATIO) & (train_df["length_ratio"] <= MAX_RATIO)
]

# Subsample to exactly 15,000 samples
TARGET_SAMPLES = 1500
if len(filtered_df) >= TARGET_SAMPLES:
    train_df = filtered_df.sample(n=TARGET_SAMPLES, random_state=42).reset_index(drop=True)
else:
    train_df = filtered_df.reset_index(drop=True)

train_data = Dataset.from_pandas(train_df)
print(f"Final Processed Training Samples: {len(train_data):,}")


# ---------------------------------------------------------------------
# Step 3: Load and Preprocess Validation Dataset (FLORES Dev)
# ---------------------------------------------------------------------
print("Loading FLORES dev dataset for validation...")
val_raw_dataset = load_dataset(
    "facebook/flores",
    "eng_Latn-mar_Deva",
    split="dev"
)
val_df = val_raw_dataset.to_pandas()

# Standardize column naming to match train data ("src" and "tgt")
val_df = val_df.rename(columns={
    "sentence_eng_Latn": "src",
    "sentence_mar_Deva": "tgt"
})
val_df = val_df[["src", "tgt"]].dropna()
val_df["src"] = val_df["src"].astype(str).str.strip()
val_df["tgt"] = val_df["tgt"].astype(str).str.strip()

val_data = Dataset.from_pandas(val_df)
print(f"Final Processed Validation Samples (FLORES Dev): {len(val_data):,}")


# ---------------------------------------------------------------------
# Step 4: Load Base Model in 4-bit Quantization
# ---------------------------------------------------------------------
print("Loading base model in 4-bit...")
use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=compute_dtype,
    attn_implementation="sdpa",
)
print(f"Model loaded. Allocated VRAM: {round(torch.cuda.memory_allocated() / 1e9, 2)} GB")


# ---------------------------------------------------------------------
# Step 5: Format Prompts and Tokenize Datasets
# ---------------------------------------------------------------------
def format_prompt(text):
    chat_history = [{
        "role": "user",
        "content": f"Translate the source English language into target Marathi language:\n\n{text}",
    }]
    return processor.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=False)

def tokenize_row(row):
    prompt_tokens = tokenizer(format_prompt(row["src"]), add_special_tokens=False)["input_ids"]
    target_tokens = tokenizer(row["tgt"], add_special_tokens=False)["input_ids"]
    
    input_ids = (prompt_tokens + target_tokens)[:MAX_LEN]
    labels = ([-100] * len(prompt_tokens) + target_tokens)[:MAX_LEN]
    
    return {
        "input_ids": input_ids, 
        "attention_mask": [1] * len(input_ids), 
        "labels": labels
    }

print("Tokenizing datasets...")
train_tokens = train_data.map(tokenize_row, remove_columns=train_data.column_names)
val_tokens = val_data.map(tokenize_row, remove_columns=val_data.column_names)

# Filter out truncated rows lacking target labels
train_tokens = train_tokens.filter(lambda x: any(l != -100 for l in x["labels"]))
val_tokens = val_tokens.filter(lambda x: any(l != -100 for l in x["labels"]))
print(f"Ready -> Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")


# ---------------------------------------------------------------------
# Step 6: Setup LoRA Adapters
# ---------------------------------------------------------------------
print("Configuring LoRA adapters...")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False

proj_types = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
target_layers = sorted(
    name for name, module in model.named_modules()
    if isinstance(module, bnb.nn.Linear4bit)
    and "language_model" in name
    and name.split(".")[-1] in proj_types
)
if not target_layers:
    target_layers = sorted(proj_types)

lora_settings = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_layers,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_settings)
model.print_trainable_parameters()


# ---------------------------------------------------------------------
# Step 7: Data Collator and Training Arguments
# ---------------------------------------------------------------------
def custom_collate_fn(batch):
    max_batch_len = max(len(item["input_ids"]) for item in batch)
    pad_token_id = tokenizer.pad_token_id
    
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        padding_needed = max_batch_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding_needed)
        attention_mask.append(item["attention_mask"] + [0] * padding_needed)
        labels.append(item["labels"] + [-100] * padding_needed)
        
    output_batch = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }
    if ADD_TOKEN_TYPE_IDS:
        output_batch["token_type_ids"] = torch.zeros_like(output_batch["input_ids"])
    return output_batch

training_config = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    optim="paged_adamw_8bit",
    bf16=use_bf16,
    fp16=not use_bf16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_config,
    train_dataset=train_tokens,
    eval_dataset=val_tokens,
    data_collator=custom_collate_fn,
)

print("Starting QLoRA fine-tuning...")
trainer.train()


# ---------------------------------------------------------------------
# Step 8: Save Model Weights to Working Directory
# ---------------------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training successfully completed! Weights saved to: {OUTPUT_DIR}")
