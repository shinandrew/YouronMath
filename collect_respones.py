import random
import operator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
import tqdm

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-math-7b-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
teacher_model.eval()

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
tokenizer.padding_side = "left"

def compute_initial_target(numbers):
    operations = [operator.add, operator.sub]
    operation_symbols = {operator.add: '+', operator.sub: '-'}
    while True:
        random.shuffle(numbers)
        ops = [random.choice(operations) for _ in range(3)]
        result = numbers[0]
        for i in range(3):
            result = ops[i](result, numbers[i + 1])
        if 0 <= result <= 100:
            return result, ops, numbers.copy()

# 100 templates
tasks = [
    "Reach {target} with {numbers}", "Get {target} using {numbers}", "Make {target} from {numbers}",
    "Combine {numbers} to get {target}", "Calculate {target} with {numbers}", "Hit {target} using {numbers}",
    "Form {target} with {numbers}", "Achieve {target} from {numbers}", "Use {numbers} to reach {target}",
    "Find {target} with foraging {numbers}"
]
operations = [
    "using only addition and subtraction", "with just + and -", "through adding and subtracting only",
    "by adding or subtracting", "using + and - exclusively", "with addition and subtraction alone",
    "through + and - operations", "by combining additions and subtractions", "using only + and -",
    "with adding and subtracting"
]
templates = [f"{task} {op}." for task in tasks for op in operations]
print(f"Generated {len(templates)} templates")

dataset_size = 250_000
batch_size = 100
all_data = {"prompt": [], "teacher_response": [], "ground_truth": []}
batch_data = {"prompt": [], "ground_truth": []} 
print_interval = 1000

print(f"Generating {dataset_size} Countdown prompts with + and - only...")
for i in tqdm.tqdm(range(dataset_size)):
    numbers = random.sample(range(1, 100), 4)
    target, correct_ops, correct_numbers = compute_initial_target(numbers)
    
    template = random.choice(templates)
    numbers_str = ", ".join(map(str, sorted(numbers)))
    prompt = template.format(numbers=numbers_str, target=target)
    
    operation_symbols = {operator.add: '+', operator.sub: '-'}
    equation = str(correct_numbers[0])
    for j in range(3):
        symbol = operation_symbols[correct_ops[j]]
        equation += f" {symbol} {correct_numbers[j + 1]}"
    ground_truth = f"{equation} = {target}"

    batch_data["prompt"].append(prompt)
    batch_data["ground_truth"].append(ground_truth)

    if len(batch_data["prompt"]) == batch_size or i == dataset_size - 1:
        with torch.no_grad():
            inputs = tokenizer(batch_data["prompt"], return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, response in zip(batch_data["prompt"], responses):
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                all_data["teacher_response"].append(response)
                all_data["prompt"].append(prompt)
                all_data["ground_truth"].append(batch_data["ground_truth"][batch_data["prompt"].index(prompt)])

        if i // batch_size % (print_interval // batch_size) == 0 or i == dataset_size - 1:
            print(f"\nBatch {i // batch_size + 1}:")
            for j in range(min(5, len(batch_data["prompt"]))):
                print(f"Prompt: {batch_data['prompt'][j]}")
                print(f"Teacher Response: {all_data['teacher_response'][-len(batch_data['prompt']) + j]}")
                print(f"Ground Truth: {batch_data['ground_truth'][j]}")
                print("-" * 50)
        
        batch_data = {"prompt": [], "ground_truth": []}  # Reset batch only
        print(f"Processed batch {i // batch_size + 1}/{dataset_size // batch_size + 1}")

# Save full dataset
dataset = Dataset.from_dict(all_data)
dataset.save_to_disk("teacher_countdown_add_sub_250k")
print("Saved 250k teacher responses to 'teacher_countdown_add_sub_250k'")

del teacher_model
torch.cuda.empty_cache()

# Print final samples
print("\nFinal Sample Outputs:")
for i in range(min(25, len(dataset))):
    print(f"Prompt: {dataset[i]['prompt']}")
    print(f"Teacher Response: {dataset[i]['teacher_response']}")
    print(f"Ground Truth: {dataset[i]['ground_truth']}")
    print("-" * 50)
