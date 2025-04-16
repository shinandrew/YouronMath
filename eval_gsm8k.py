import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import re
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "Qwen/Qwen2.5-Math-1.5B"
model_path = "./outputs/GRPO/qa/Qwen/Qwen2.5-Math-1.5B"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, model_path)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = " PAD_TOKEN"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(" PAD_TOKEN")
model.config.pad_token_id = tokenizer.pad_token_id
print("Fine-tuned model and tokenizer loaded with pad token set.")

gsm8k_train = load_dataset("gsm8k", "main", split="train")
gsm8k_test = load_dataset("gsm8k", "main", split="test")
print(f"Loaded GSM8K: {len(gsm8k_train)} train samples, {len(gsm8k_test)} test samples.")

# Select 8-shot examples from training set
random.seed(42)
few_shot_examples = random.sample(list(gsm8k_train), 8)

# Build 8-shot prompt
def build_few_shot_prompt():
    prompt = ""
    for example in few_shot_examples:
        question = example["question"]
        answer = example["answer"]
        if "####" in answer:
            reasoning = answer.split("####")[0].strip()
            final_answer = answer.split("####")[-1].strip()
        else:
            reasoning = answer.strip()
            final_answer = ""
        prompt += f"Question: {question}\nSolution: Let's think step by step. {reasoning}\n#### The final answer is {final_answer}\n\n"
    return prompt

few_shot_prompt = build_few_shot_prompt()
print("8-shot prompt built.")

def extract_number(response):
    match = re.search(r'#### The final answer is (-?\d+\.?\d*)', response)
    if match:
        return float(match.group(1))
    match = re.search(r'(?:answer is|final answer:?\s+)(\d+\.?\d*)', response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    matches = re.findall(r'-?\d+\.?\d*', response)
    return float(matches[-1]) if matches else None

def evaluate_model(test_dataset, max_samples=None):
    correct_count = 0
    total_count = 0
    sample_outputs = []

    eval_dataset = test_dataset if max_samples is None else test_dataset.select(range(min(max_samples, len(test_dataset))))

    for example in tqdm(eval_dataset, desc="Evaluating"):
        question = example["question"]
        ground_truth_answer = example["answer"]
        gt_match = re.search(r'-?\d+\.?\d*', ground_truth_answer.split("####")[-1].strip())
        ground_truth = float(gt_match.group()) if gt_match else None

        prompt = f"{few_shot_prompt}Question: {question}\nSolution: Let's think step by step. "

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            padding_side="left",
            return_attention_mask=True
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_token_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_token_ids = output_ids[0, prompt_token_len:].tolist()
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True) if new_token_ids else ""

        # Extract predicted answer
        predicted = extract_number(response)

        is_correct = predicted is not None and ground_truth is not None and abs(predicted - ground_truth) < 0.1
        if is_correct:
            correct_count += 1
        total_count += 1

        if total_count <= 10:
            sample_outputs.append({
                "question": question,
                "full_output": full_output,
                "response": response,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "input_tokens": prompt_token_len,
                "output_tokens": output_ids.shape[1]
            })

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, correct_count, total_count, sample_outputs

max_eval_samples = None
accuracy, correct, total, sample_outputs = evaluate_model(gsm8k_test, max_samples=max_eval_samples)

# Print results
print(f"\nEvaluation on GSM8K test set ({total} samples):")
print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

torch.cuda.empty_cache()
