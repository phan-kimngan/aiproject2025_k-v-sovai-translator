from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
import torch, json, pandas as pd, re
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import torch, gc
torch.cuda.empty_cache()
gc.collect()

model_path = "./skt-ax3.1light-vie-kor-qlora"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    offload_folder="./offload",        # 💡 thêm dòng này
    low_cpu_mem_usage=True             # tuỳ chọn
)

model.config.pad_token_id = model.config.eos_token_id
model.eval()


dataset = DatasetDict({
    "train": load_dataset("json", data_files="Advanced AI Project/train.json")["train"],
    "val":   load_dataset("json", data_files="Advanced AI Project/val.json")["train"],
    "test":  load_dataset("json", data_files="Advanced AI Project/test.json")["train"],
})
test_data = dataset["test"]

batch_size = 8
results = []

for i in tqdm(range(0, len(test_data), batch_size), desc="Generating translations (clean)"):
    batch = test_data[i:i + batch_size]
    batch_vie = batch["output"]
    batch_refs = batch["input"]

    # Tạo prompt đúng chuẩn LLaMA-3
    batch_prompts = []
    for text in batch_vie:
        messages = [
            {"role": "system", "content": "You are a helpful translator that translates Vietnamese to Korean."},
            {"role": "user", "content": f"Translate the following Vietnamese sentence into Korean:\n\n{text}"},
        ]
        batch_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )

    # Tokenize
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.3,
            top_p=0.8,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    attn = inputs["attention_mask"]
    input_lengths = attn.sum(dim=1).tolist()

    for seq, in_len, src, ref in zip(outputs, input_lengths, batch_vie, batch_refs):
        gen_ids = seq[in_len:]    
        

        import unicodedata

        generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        

        generated = unicodedata.normalize("NFKC", generated)


        generated = re.sub(
            r"(?is)(translate\s+the\s+following\s+korean\s+sentence\s+into\s+vietnamese[:：]?\s*|"
            r"you\s+are\s+a\s+helpful\s+translator\s+that\s+translates[:：]?\s*|"
            r"are\s+a\s+helpful\s+translator\s+that\s+translates[:：]?\s*|"
            r"that\s+translates[:：]?\s*|"
            r"to\s+vietnamese[:：]?\s*|"
            r"vietnamese[:：]?\s*|"
            r"sentence\s+in[:：]?\s*|"
            r"in[:：]?\s*|"
            r"following\s+korean\s+sentence\s+into\s+vietnamese[:：]?\s*|"
            r"user[:：]?|assistant[:：]?|input[:：]?|output[:：]?|korean[:：]?|english[:：]?|the\s+)?",
            "",
            generated,
        )


        generated = generated.replace("\ufeff", "")
        generated = re.sub(r"[\u200B-\u200D\uFEFF\u00A0]+", "", generated)  # zero-width, NBSP

        generated = re.sub(r"[A-Za-zÀ-ỿà-ỹĂăÂâĐđÊêÔôƠơƯư�]+", "", generated)

        generated = re.sub(r"[\r\n\t]+", " ", generated)
        generated = re.sub(r"\s+", " ", generated).strip()


        patterns_head = [

            r"^\s*(?:\d+(?:\s*[,，]\s*\d+)*)(?:[\.\u3002,，:：;\)\]-–—])\s*",

            r"^(?:\d+(?:[\.\u3002,，:：;\)\]-–—])\s*)+",

            r"^\s*\d+[\.\u3002:：;\)\]-–—]+",

            r"^[\.\u3002,，:：;!\?…·•\(\)\[\]\{\}\-–—\"“”\'\s]+",

            r"^(?:korean|the|english|sentence|input|output|are|a|helpful|translator|that|translates)\b[\s\.\,:\-–—;!\?\"“”\'\)]*",
        ]

        for _ in range(5):
            old = generated
            for pat in patterns_head:
                generated = re.sub(pat, "", generated, flags=re.IGNORECASE)
            generated = generated.lstrip()
            if generated == old:
                break


        generated = re.sub(r"\s+", " ", generated).strip()

        if generated:
            generated = generated[0].upper() + generated[1:]

        

        print(f"src: {src}")
        print(f"reference_korean: {ref}")
        print(f"predicted_korean: {generated}")
        print("\n")
        results.append({
            "vietnamese": src,
            "reference_korean": ref,
            "predicted_korean": generated
        })


output_json = "test_vie2kor.json"
output_csv = "test_vie2kor.csv"

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\nĐã lưu kết quả inference:")
print(f"   - JSON: {output_json}")
print(f"   - CSV:  {output_csv}")



