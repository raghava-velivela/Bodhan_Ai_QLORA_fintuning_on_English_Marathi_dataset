import json
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
import evaluate

# ---------------------------------------------------------------------
# Step 1: Load IN22-Conv Test Dataset (Restricted to 50 Samples for Safety)
# ---------------------------------------------------------------------
print("Loading IN22-Conv test dataset...")
dataset = load_dataset("ai4bharat/IN22-Conv", split="test")

test_df = dataset.select_columns(["eng_Latn", "mar_Deva"]).to_pandas()
test_df = test_df.rename(columns={
    "eng_Latn": "src",
    "mar_Deva": "tgt"
})
test_df = test_df[["src", "tgt"]].dropna()
test_df["src"] = test_df["src"].astype(str).str.strip()
test_df["tgt"] = test_df["tgt"].astype(str).str.strip()

# Restrict to 50 samples to prevent VRAM spikes during generation
test_df = test_df.head(50).reset_index(drop=True)
print(f"Test samples ready for evaluation: {len(test_df):,}")


# ---------------------------------------------------------------------
# Step 2: Load Processor and Base Model in 4-bit (Prevents OOM)
# ---------------------------------------------------------------------
BASE_MODEL_ID = "bodhan-ai/indic-translate"
LORA_WEIGHTS_DIR = "/kaggle/working/indic-translate-marathi-lora"
MAX_NEW_TOKENS = 128
OUTPUT_JSONL = "/kaggle/working/predictions.jsonl"

print("Loading processor from base model...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
tokenizer = processor.tokenizer
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

torch.cuda.empty_cache()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model in 4-bit...")
base_model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=compute_dtype,
    attn_implementation="sdpa",
)

print(f"Loading LoRA adapters from: {LORA_WEIGHTS_DIR}")
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_DIR)
model.eval()
model.config.use_cache = True


# ---------------------------------------------------------------------
# Step 3: Define Prompt & Translation Generation Function
# ---------------------------------------------------------------------
def format_prompt(text):
    chat_history = [{
        "role": "user",
        "content": f"Translate the source English language into target Marathi language:\n\n{text}",
    }]
    return processor.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=False)

def translate_sentence(text):
    prompt = format_prompt(text)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True
        )
    
    generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------
# Step 4: Run Inference & Export JSONL
# ---------------------------------------------------------------------
print("\nRunning model generation on test samples...")
predictions = []
references = []
jsonl_records = []

for idx, row in test_df.iterrows():
    src_text = row["src"]
    ref_text = row["tgt"]
    
    pred_text = translate_sentence(src_text)
    
    predictions.append(pred_text)
    references.append([ref_text])
    
    record = {
        "id": idx,
        "source_en": src_text,
        "reference_mr": ref_text,
        "prediction_mr": pred_text
    }
    jsonl_records.append(record)
    
    if idx < 5:
        print(f"\n[Sample {idx+1}]")
        print(f"English (Source)    : {src_text}")
        print(f"Reference (Target)  : {ref_text}")
        print(f"Model Prediction    : {pred_text}")
        print("-" * 50)

print(f"\nExporting results to {OUTPUT_JSONL}...")
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for record in jsonl_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print("Export complete!")


# ---------------------------------------------------------------------
# Step 5: Compute chrF++ Metric Score
# ---------------------------------------------------------------------
print("\nCalculating chrF++ evaluation score...")
chrf_metric = evaluate.load("chrf")

results = chrf_metric.compute(
    predictions=predictions,
    references=references,
    word_order=2
)

print("=" * 60)
print(f"FINAL EVALUATION REPORT")
print(f"chrF++ Score: {results['score']:.2f}")
print(f"Predictions saved to: {OUTPUT_JSONL}")
print("=" * 60)
