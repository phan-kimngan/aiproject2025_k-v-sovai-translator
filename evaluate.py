import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
### For Korean to Vietnamese 
file_path = "Advanced AI Project/test_kor2vie.json"
with open(file_path, "r", encoding="utf-8") as f:
    results = json.load(f)


valid_results = [
    r for r in results 
    if "reference_vietnamese" in r and "predicted_vietnamese" in r 
       and isinstance(r["reference_vietnamese"], str)
       and isinstance(r["predicted_vietnamese"], str)
       and r["reference_vietnamese"].strip() and r["predicted_vietnamese"].strip()
]

refs = [[r["reference_vietnamese"].split()] for r in valid_results[10008:]]
hyps = [r["predicted_vietnamese"].split() for r in valid_results[10008:]]

smooth = SmoothingFunction().method1

bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth)
bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
bleu3 = corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

meteor = sum(
    meteor_score(
        [r["reference_vietnamese"].split()],
        r["predicted_vietnamese"].split()
    )
    for r in valid_results[10008:]
) / len(valid_results[10008:])

print("\n🎯 Evaluation metrics for Korean to Vietnamese  :")
print(f"BLEU-1:  {bleu1 * 100:.2f}")
print(f"BLEU-2:  {bleu2 * 100:.2f}")
print(f"BLEU-3:  {bleu3 * 100:.2f}")
print(f"BLEU-4:  {bleu4 * 100:.2f}")
print(f"METEOR:  {meteor * 100:.2f}")


### For Vietnamese to Korean 
file_path = "Advanced AI Project/test_vie2kor.json"
with open(file_path, "r", encoding="utf-8") as f:
    results = json.load(f)
    
    
valid_results = [
    r for r in results 
    if "reference_korean" in r and "predicted_korean" in r 
       and isinstance(r["reference_korean"], str)
       and isinstance(r["predicted_korean"], str)
       and r["reference_korean"].strip() and r["predicted_korean"].strip()
]

refs = [[r["reference_korean"].split()] for r in valid_results[10008:]]
hyps = [r["predicted_korean"].split() for r in valid_results[10008:]]

smooth = SmoothingFunction().method1

bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth)
bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
bleu3 = corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

meteor = sum(
    meteor_score(
        [r["reference_korean"].split()],
        r["predicted_korean"].split()
    )
    for r in valid_results[10008:]
) / len(valid_results[10008:])


print("\n🎯 Evaluation metrics for Vietnamese to Korean :")
print(f"BLEU-1:  {bleu1 * 100:.2f}")
print(f"BLEU-2:  {bleu2 * 100:.2f}")
print(f"BLEU-3:  {bleu3 * 100:.2f}")
print(f"BLEU-4:  {bleu4 * 100:.2f}")
print(f"METEOR:  {meteor * 100:.2f}")
print(f"\n📊 Tổng số mẫu đánh giá: {len(valid_results)}")
