# gsm8k.py

import random
import json
from resources import EXAMPLARS
from datasets import load_dataset

class GSM8K:
    def __init__(self, split, include_answer=True, include_reasoning=True, few_shot=False, num_shots=8, seed=None, cot=False, template="qa"):
        self.split = split
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)

        self.few_shot = few_shot 
        self.num_shots = num_shots
        self.cot = cot
        self.template = template
        self.examples = None
        self.dataset = self.load_dataset()
        
    def format_example(self, question, solution, answer):
        example = ''
  
        if self.template == 'qa':
            example = f"Question: {question}\nSolution: "
            
            if self.cot:
                example += "Let's think step by step. "
                        
            if solution is not None:
                def remove_placeholders(text):
                    import re
                    # Regex to match <<anything>>
                    cleaned_text = re.sub(r'<<.*?>>', '', text)
                    return cleaned_text

                solution = '. '.join(solution.split('\n'))
                solution = remove_placeholders(solution)
                example += f"{solution}.\n"
            
            example = example.replace('..', '.')
            
            if answer is not None:
                example += f"#### The final answer is {answer}\n\n"
                
        elif self.template == 'code':
            example = f'Question: {question}\n\n# solution in Python:\n\ndef solution():\n    """{question}"""\n'
            
            if solution is not None:
                example += f'{solution}\n\n'
        else:
            raise ValueError('Format Not Implemented')
        
        return example  
    
    def process_example(self, example, index):
        
        question = example['question']
        answer = example['answer']
        
        # Extract the reasoning steps and the final answer
        answer_delim = "#### "
        if answer_delim in answer:
            reasoning = answer.split(answer_delim)[0].strip()
            final_answer = answer.split(answer_delim)[-1].strip()
        else:
            reasoning = answer.strip()
            final_answer = ''
            
        # Create the prompt
        if self.include_answer:
            if self.include_reasoning:
                input_text = self.format_example(question, reasoning, final_answer)
            else:
                input_text = self.format_example(question, None, final_answer)
        else:
            input_text = self.format_example(question, None, None)

        if self.few_shot:
            input_text = self.few_shot_prompt + input_text

        print("INPUTTEXT:",input_text)
        print("FINALANSWER:",final_answer)
        print("QUESTION:",question)
        print('=====')
        return {
            'prompt': input_text,
            'final_answer': final_answer,
            'question': question, 
        }    

    def load_dataset(self):
        # Load the GSM8K dataset with the specified split
        dataset = load_dataset('gsm8k', 'main', split=self.split)
        
        if self.few_shot:
            self.few_shot_prompt = self.build_prompt()

        dataset = dataset.map(self.process_example, with_indices=True, load_from_cache_file=False)

        return dataset
    
    
    def fewshot_examples_code(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "resources/gsm8k_few_shot_prompts.json",
            "r",
        ) as file:
            examples = json.load(file)
        
        new_examples = []
        for question, solution in zip(
            examples["questions"][:self.num_shots], examples["solutions"][:self.num_shots]
        ):
           new_examples.append(dict(
                question=question,
                cot_answer=solution,
                short_answer=None
            ))

            
        return new_examples
    
    def fewshot_examples_qa(self):        
        return EXAMPLARS

    def make_prompts(self):
        """Builds the prompt for the LM to generate from."""

        if self.template == 'qa':
            examples = self.fewshot_examples_qa()
        elif self.template == 'code':
            examples = self.fewshot_examples_code()
        else:
            raise ValueError('Format Not Implemented')
    
        self.examples = examples        
    
    def build_prompt(self):
        
        if self.examples is None:
            self.make_prompts()
                
        prompt = ""
        for qna in random.sample(self.examples, self.num_shots):
            prompt += self.format_example(qna['question'], qna['cot_answer'], qna['short_answer'])
            
        return prompt
