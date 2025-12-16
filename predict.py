from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, unicodedata
# --- Load base model ---
model_path_kor_vie = "./skt-ax3.1light-kor-vie-qlora"
model_kor_vie = AutoModelForCausalLM.from_pretrained(
model_path_kor_vie,
torch_dtype=torch.bfloat16,   # ✅ use standard 32-bit float for CPU  
device_map="auto",              # ✅ disable auto mapping to CUDA,
)
model_kor_vie.eval()
# --- Tokenizer ---
tokenizer_kor_vie = AutoTokenizer.from_pretrained(model_path_kor_vie)
tokenizer_kor_vie.pad_token = tokenizer_kor_vie.eos_token
tokenizer_kor_vie.padding_side = "left"

def preprocesing(generated, task = None):
# 1) Gỡ prompt rác/instruction lặp
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
    if task =="kor_vie":
        generated = re.sub(r"[가-힣ㄱ-ㅎㅏ-ㅣ�]+", "", generated)            # nếu còn sót Hàn + ký tự lỗi
    if task =="vie_kor":
        generated = re.sub(r"[A-Za-zÀ-ỿà-ỹĂăÂâĐđÊêÔôƠơƯư�]+", "", generated)

    # 3) Thu gọn whitespace
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

    # 5) Dọn lại khoảng trắng cuối cùng (không đụng dấu câu cuối)
    generated = re.sub(r"\s+", " ", generated).strip()

    # 6) Viết hoa chữ cái đầu nếu có
    if generated:
        generated = generated[0].upper() + generated[1:]
    return generated
def translate_kor_to_vie(text: str, max_new_tokens=80) -> str:
    """Dịch từ tiếng Hàn sang tiếng Việt bằng mô hình đã fine-tune"""
    if not text.strip():
        return ""

    messages = [
        {"role": "system", "content": "You are a helpful translator that translates Korean to Vietnamese."},
        {"role": "user", "content": f"Translate the following Korean sentence into Vietnamese:\n\n{text}"},
    ]
    prompt = tokenizer_kor_vie.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer_kor_vie(prompt, return_tensors="pt", padding=True, truncation=True).to(model_kor_vie.device)

    with torch.no_grad():
        outputs = model_kor_vie.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.8,            
            do_sample=True,
            eos_token_id=tokenizer_kor_vie.eos_token_id,
            pad_token_id=tokenizer_kor_vie.eos_token_id,
        )

    input_len = inputs["attention_mask"].sum(dim=1).item()
    gen_ids = outputs[0][input_len:]
    generated = tokenizer_kor_vie.decode(gen_ids, skip_special_tokens=True).strip()

    # --- Làm sạch kết quả ---
    generated = unicodedata.normalize("NFKC", generated)
    generated = preprocesing(generated, task = "kor_vie")
    #generated = re.sub(r"\s+", " ", generated).strip()
    return generated

