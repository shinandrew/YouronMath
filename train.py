import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from utils import format_reward_func_qa, correctness_reward_func_qa, \
                  format_reward_func_code, correctness_reward_func_code, \
                  print_trainable_parameters
from gsm8k import GSM8K

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for Reasoning on GSM8K with RLVR and GRPO.")
    parser.add_argument('--format', type=str, default='qa', choices=['qa', 'code'])
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Math-1.5B')
    return parser.parse_args()

args = parse_args()
print(args)

dataset = GSM8K(split='train', include_answer=False, include_reasoning=True, few_shot=True, num_shots=args.num_shots, seed=None, cot=True, template=args.format).dataset.shuffle(seed=42)

model_name = args.model_name
output_dir = f'outputs/GRPO/{args.format}/{model_name}'

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=f'GRPO-GSM8K-{args.format}-{model_name.split('/')[-1]}',
    learning_rate=2e-5,
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=128,
    max_completion_length=150,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to='wandb',
    log_on_each_node=False,
    # use_vllm=True,
    # vllm_device='auto',
)


rank = 16
peft_config = LoraConfig(
    r=rank,
    lora_alpha=rank*2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    bias='none',
    lora_dropout=0.05,
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,#bfloat16,
    attn_implementation="flash_attention_2",
    device_map='auto'
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

rewards_funcs = []

if args.format == 'qa':
    rewards_funcs = [format_reward_func_qa, correctness_reward_func_qa]
elif args.format == 'code':
    rewards_funcs = [format_reward_func_code, correctness_reward_func_code]

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rewards_funcs,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(output_dir)
print(f"LoRA model and configuration saved to {output_dir}")
