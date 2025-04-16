import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
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
print(f"GSM8K fine-tuned model loaded from {model_path}.")

# Load MMLU-STEM test dataset (math only)
dataset_name = "cais/mmlu"
math_configs = [
    "college_mathematics",
    "high_school_mathematics"
]
test_datasets = [load_dataset(dataset_name, config, split="test") for config in math_configs]
test_dataset = concatenate_datasets(test_datasets)
print(f"Loaded {dataset_name} (Math only) test: {len(test_dataset)} samples.")

dev_datasets = [load_dataset(dataset_name, config, split="dev") for config in math_configs]
dev_dataset = concatenate_datasets(dev_datasets)
random.seed(42)
few_shot_examples = random.sample(list(dev_dataset), 4)
print(f"Loaded {dataset_name} (Math only) dev for 4-shot: {len(dev_dataset)} samples.")

# Build 4-shot prompt (with choices)
def build_few_shot_prompt():
    prompt = ""
    for example in few_shot_examples:
        question = example["question"]
        choices = example["choices"]
        correct_idx = example["answer"]
        choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
        answer = f"Let's think step by step. The correct answer is {choices[correct_idx]}.\n#### The final answer is {correct_idx}"
        prompt += f"Question: {question}\nChoices:\n{choices_str}\nSolution: Let's think step by step. {answer}\n\n"
    return prompt

few_shot_prompt = build_few_shot_prompt()
print("4-shot prompt built.")

# Extract answer (multiple-choice)
def extract_answer(response, choices):
    match = re.search(r'#### The final answer is\s*([0-3])', response)
    if match:
        return int(match.group(1))
    
    match = re.search(r'(?:answer is|final answer:?\s*)([A-D])', response, re.IGNORECASE)
    if match:
        return ord(match.group(1).upper()) - ord('A')
    
    matches = re.findall(r'[0-3]', response)
    if matches:
        return int(matches[-1])
    
    for i, choice in enumerate(choices):
        if choice.lower() in response.lower():
            return i
    
    return None

def evaluate_model(test_dataset, max_samples=None):
    correct_count = 0
    total_count = 0
    sample_outputs = []

    eval_dataset = test_dataset if max_samples is None else test_dataset.select(range(min(max_samples, len(test_dataset))))

    for example in tqdm(eval_dataset, desc="Evaluating"):
        question = example["question"]
        choices = example["choices"]
        ground_truth = int(example["answer"])
        choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])

        prompt = f"{few_shot_prompt}Question: {question}\nChoices:\n{choices_str}\nSolution: Let's think step by step. "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True, padding_side="left", return_attention_mask=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_token_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,        # Sampling per paper
                temperature=0.7,       # Moderate temperature
                top_k=50,             # Top-k sampling
                pad_token_id=tokenizer.pad_token_id,
            )
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_token_ids = output_ids[0, prompt_token_len:].tolist()
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True) if new_token_ids else ""

        predicted = extract_answer(response, choices)
        is_correct = predicted is not None and predicted == ground_truth
        if is_correct:
            correct_count += 1
        total_count += 1

        if total_count <= 10:
            sample_outputs.append({
                "question": question,
                "choices": choices,
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

max_eval_samples = None  # Full math-only test set
accuracy, correct, total, sample_outputs = evaluate_model(test_dataset, max_samples=max_eval_samples)

print(f"\nEvaluation of GSM8K fine-tuned {model_name} on MMLU-STEM (Math only) test set ({total} samples):")
print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

torch.cuda.empty_cache()
