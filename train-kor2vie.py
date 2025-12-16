from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

# ===================================
# ✅ 1. Model: SKT A.X-3.1-Light (7B)
# ===================================
model_name = "skt/A.X-3.1-Light"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,          # ✅ Prevent tokenizer crash
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===================================
# ✅ 2. Load dataset & apply chat template
# ===================================
dataset = DatasetDict({
    "train": load_dataset("json", data_files="Advanced AI Project/train.json")["train"],
    "val":   load_dataset("json", data_files="Advanced AI Project/val.json")["train"],
    "test":  load_dataset("json", data_files="Advanced AI Project/test.json")["train"],
})

def format_example(ex):
    user_msg = f"Translate the following Vietnamese sentence into Korean.\n\nInput:\n{ex['output']}" if ex.get("output") else "Translate the following Korean sentence into Vietnamese."
    messages = [
        {"role": "system", "content": "You are a helpful translator that translates Vietnamese to Korean."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ex["input"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }

dataset = dataset.map(format_example)
dataset["train"] = dataset["train"].select(range(len(dataset["train"]) // 2))  # Optional speed-up
dataset["val"] = dataset["val"].select(range(len(dataset["val"]) // 2))

# ===================================
# ✅ 3. QLoRA Config (4-bit quantization)
# ===================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

print("✅ GPU available:", torch.cuda.is_available())
print("✅ Running on:", next(model.parameters()).device)

# ===================================
# ✅ 4. LoRA Config (for SKT A.X architecture)
# ===================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# ===================================
# ✅ 5. Training Settings
# ===================================
sft_config = SFTConfig(
    output_dir="./skt-ax3.1light-vie-kor-qlora",
    dataset_text_field="text",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
    packing=False,
)

# ✅ IMPORTANT: Add tokenizer to prevent FastTokenizer reload crash
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,   # ✅ Fix inside TRL
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
)

# ===================================
# ✅ 6. Train & Save
# ===================================
trainer.train()

model.save_pretrained("./skt-ax3.1light-vie-kor-qlora")
tokenizer.save_pretrained("./skt-ax3.1light-vie-kor-qlora")

print("✅ Training done successfully for skt/A.X-3.1-Light!")

