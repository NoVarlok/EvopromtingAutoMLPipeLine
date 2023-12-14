"""
    Giving an AI the ability to improve its own underlying architecture through an evolutionary algorithm is, besides being poetically 
    beautiful, also a very promising paradigm. This paper is heavily based on the EvoPrompt paper by Angelica Chen David M. Dohan and David R. So.
    The original soft promted tuned a PALM 62B model. Since I dont have access to this model i instead finetune gpt3. Which is an expensive endeavour, but 
    very cool nevertheless. 
"""

import concurrent.futures
import json
import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F

from openai import OpenAI

from utils import Config, load_config

class EvoPrompting:
    def __init__(self, seed_folder,
                 train_dataset, test_dataset, metric_fn, loss_fn, device,
                 T, m, k, n, p, alpha, n_evaluations,
                 target_model_size, target_metric_score, task,
                 openai_model, openai_token,
                 incorrect_models_dir,
                 seed_evaluation=False, evaluation_path=None):
        self.seed_folder = seed_folder # Folder where the seed codes are located
        self.seed_evaluation = seed_evaluation # Do we have to evaluate the seed codes?
        self.pre_evaluated_seed_metrics = self.load_pre_evaluated_seed_metrics(evaluation_path) # Pre evaluated seed metrics
        self.temperatures = [0.2, 0.6, 0.8, 1.0] # uniformly sample from these temperaturs
        self.train_dataset = train_dataset
        self.test_datsaet = test_dataset
        self.metric_fn = metric_fn
        self.loss_fn = loss_fn
        self.device = device
        self.T = T # Number of rounds
        self.m = m # number of few-shot prompts per round
        self.n = n # number of samples to generate per prompt,
        self.k = k # number of in-context examples per prompt
        self.p = p # number of survivors to select per generation
        self.n_evaluations = n_evaluations # Number of times to run each model
        self.alpha = alpha # the upper threshold for the test error
        self.global_population = [] # Global historical Population

        self.target_model_size = target_model_size # Target model size of the few shot prompt
        self.target_metric_score = target_metric_score # Target number of episodes of the few shot prompt
        self.task = task

        self.openai_model = openai_model
        self.client = OpenAI(api_key=openai_token)
        
        # Set initial well designed architectures as parent models.
        # (Evaluate them useing the same eval function as used in the aalgo)
        self.current_population = []
        self.training_code = self.read_seed_files(os.path.join(self.seed_folder, 'main.py'))
        self.incorrect_models_dir = incorrect_models_dir
        self.initialize_population()
    

    def read_seed_files(self, file_path):
        with open(file_path, "r") as file:
            return file.read()


    def load_pre_evaluated_seed_metrics(self, file_path):
        with open(file_path, "r") as file:
            return json.load(file)


    def initialize_population(self):
        # Initialize the population with seed architectures
        # List all the Python files in the seed folder
        seed_files = [f for f in os.listdir(self.seed_folder) if f.endswith('.py') and f != 'main.py']
        # seed_files = ['simple_mlp_layer.py']

        for seed_file in seed_files:
            print("EVALUATING SEED: ", seed_file)
            seed_file_path = os.path.join(self.seed_folder, seed_file)
            seed_code = self.read_seed_files(seed_file_path)

            if self.seed_evaluation:
                avg_metric, model_size = self.eval_t(seed_code)
            else:
                json= self.pre_evaluated_seed_metrics[seed_file]
                # convert string to float           
                avg_metric = float(json["avg_metric"])
                model_size = float(json["model_size"])

            print("EVALUATED SEED: ", seed_file, "avg_metric: ", avg_metric, "model_size: ", model_size)
            metrics = {
                "avg_metric": avg_metric,
                "model_size": model_size,
            }
            
            # fitness_score = avg_metric * model_size
            fitness_score = self.fitness_function(model_size, avg_metric)
            self.global_population.append((seed_code, metrics, fitness_score))
            self.current_population.append((seed_code, metrics, fitness_score))
        

    def make_few_shot_prompt(self, in_context_examples):
        # Create a few-shot prompt using the in context examples E
        min_avg_metric = float('inf')
        min_model_size = float('inf')
        prompt = "" # Initialize empty prompt string

        for example in in_context_examples:
            metrics = example[1]
            min_avg_metric = min(min_avg_metric, metrics['avg_metric']) # Retrieve the minium avg metric of the parent architectures
            min_model_size = min(min_model_size, metrics['model_size']) # Retrieve the minium model size of the parent architectures
            prompt += f'\nMetrics: {example[1]}\n\n'
            prompt += f'Code: {example[0]}\n\n'

        target_avg = min_avg_metric * self.target_metric_score
        target_model_size = min_model_size * self.target_model_size

        prompt += f'\nmetrics: {{ "avg_metric": {target_avg}, "model_size": {target_model_size} }}\n\n'
        # prompt += f'Code:\n'
        prompt += f'Generate model architecture based on mutation of these models, that reachers better metrics value. Do not generate training code\nCode:'
        # print(prompt)

        return prompt


    def generate_child (self, prompt):
        child_code = self.client.chat.completions.create(
            # model="gpt-3.5-turbo-1106",
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=np.random.choice(self.temperatures, size=1, replace=True).item(),
            n=1,
            max_tokens = 1000
        )
        return child_code.choices[0].message.content


    def eval_t(self, code_segment):
        def single_evaluation():
            try:
                print("Executing code segment")
                training_code_segment = code_segment + '\n\n' + self.training_code + '\n'
                exec(training_code_segment, globals())  # Add globals() here
                metric, model_size = globals()['main'](self.train_dataset, self.test_datsaet, self.metric_fn, self.loss_fn, self.device)
                print(f"Finished executing code segment: metric={metric}, model_size={model_size}")
                return metric, model_size
            except Exception as e:
                print('Exception:', e)
                filename = f'{len(os.listdir(self.incorrect_models_dir))}.py'
                with open(os.path.join(self.incorrect_models_dir, filename), 'w') as f:
                    print(code_segment, file=f)
                return np.inf, np.inf

        sum_metric = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("Submitting tasks to the thread pool")
            futures = [executor.submit(single_evaluation) for _ in range(self.n_evaluations)]
            for future in concurrent.futures.as_completed(futures):
                metric, model_size = future.result()
                sum_metric += metric

        avg_metric = sum_metric / self.n_evaluations
        print(f"Average metric: {avg_metric}, Model size: {model_size}")
        return avg_metric, model_size


    def get_top(self, global_population):
        """
        Returns the top entries from the global_population based on their fitness scores.

        This function takes a list of global_population entries, where each entry is a tuple containing:
        (code, metadata, fitness_score). It sorts the entries based on their fitness scores in descending
        order and returns the top num_top entries.

        Parameters:
        global_population (list): A list of tuples, where each tuple represents an entry in the global
                                population, containing (code, metadata, fitness_score).
        num_top (int, optional): The number of top entries to return. Defaults to 5.

        Returns:
        list: A list containing the top num_top entries from the global_population based on their fitness
            scores.
        """
        sorted_population = sorted(global_population, key=lambda x: x[2], reverse=False)
        print('Sorted Scores:', [x[2] for x in sorted_population])
        top_entries = sorted_population[:self.p]
        return top_entries


    def cross_mutation(self):
        child_architectures = [] # C is the set of architectures of length k
        for _ in range(self.m): # create m number of few shot prompts
            in_context_examples = random.sample(self.current_population, self.k) # Pick k amount of parants from P
            prompt = self.make_few_shot_prompt(in_context_examples)
            Ci = [self.generate_child(prompt) for _ in range(self.n)]
            child_architectures.extend(Ci)
        return child_architectures


    def fitness_function(self, model_size, metric):
        if np.isinf(metric):
            return np.inf
        if self.task == "classification":
            target_metric = 1 - metric
        else:
            target_metric = metric
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        return sigmoid(model_size) * target_metric


    def filter_and_eval(self, child_architectures, alpha):
        CEVALED = []
        for code_segment in child_architectures:
            avg_metric, model_size = self.eval_t(code_segment)
            # if avg_metric < alpha: # filter out the bad models
            if True:
                metrics = {
                    "avg_metric": avg_metric,
                    "model_size": model_size,
                }
                fitness_score = self.fitness_function(model_size, avg_metric)
                CEVALED.append((code_segment, metrics, fitness_score))
        return CEVALED
    

    def train(self, CEVALED):
        # The original author of the paper proposes a soft prompt tune method here
        # I need a model here that can be soft promt tuned, probably gpt2 on huggingface.
        pass

    def evolve(self):
        t = 0
        while t < self.T: # number of evoluationary rounds
            print('=' * 50)
            print('Evolution round:', t)
            print('=' * 50)
            child_architectures = self.cross_mutation() # Generate the set of code samples
            evaluated_children = self.filter_and_eval(child_architectures, self.alpha)
            self.global_population.extend(evaluated_children)

            if t < self.T - 1:
                self.current_population = self.get_top(global_population=self.global_population)
                self.global_population = self.current_population
                #run without training
                #self.lm = self.train(self.lm, [c for c, _ in evaluated_children if c not in self.current_population])
            
            t += 1 

        return self.get_top(global_population=self.global_population)
    
    def save_results(self, directory):
        top_k = self.get_top(global_population=self.global_population)
        results = {
            f'model_{i}': {
                'metadata': data[1],
                'fitness_score': data[2],
            } for i, data in enumerate(top_k)
        }
        json_path = os.path.join(directory, 'results.json')
        with open(json_path, 'w') as fout:
            fout.write(json.dumps(results, indent=4))
        for i, data in enumerate(top_k):
            model_name = f'generated_model_{i}.py'
            model_path = os.path.join(directory, model_name)
            with open(model_path, 'w') as fout:
                print(data[0], file=fout)


def prepare_data_tensor(csv_path, target_name, batch_size, target_type):
    train = pd.read_csv(csv_path)
    train_target = torch.tensor(train[target_name].values.astype(target_type))
    train = torch.tensor(train.drop(target_name, axis = 1).values.astype(np.float32)) 
    train_tensor = data_utils.TensorDataset(train, train_target) 
    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
    return train_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config-path', type=str, required=True, help='Path to config')
    args = parser.parse_args()
    
    config = load_config(args.config_path)

    os.makedirs(config.save_directory, exist_ok=True)
    os.makedirs(config.incorrect_models_dir, exist_ok=True)

    target_type = np.int64 if config.dataset_type == 'classification' else np.float32
    train_dataloader = prepare_data_tensor(config.train_dataset_path, config.target_name, config.batch_size, target_type)
    test_dataloader = prepare_data_tensor(config.test_dataset_path, config.target_name, config.batch_size, target_type)

    if config.dataset_type == 'classification':
        metric_fn = lambda y_true, y_predicted: (y_true == y_predicted).sum()
        loss_fn = F.cross_entropy
    else:
        metric_fn = F.mse_loss
        loss_fn = F.mse_loss

    evo_prompt = EvoPrompting(config.seed_folder,
                              train_dataloader, test_dataloader, metric_fn, loss_fn, config.device,
                              config.rounds_count, config.few_shot_prompts_per_round,
                              config.in_context_examples_per_prompt, config.samples_per_prompt,
                              config.survivors_per_generation, config.alpha, config.n_evaluations,
                              config.target_model_factor, config.target_metric, config.dataset_type,
                              config.openai_model_name, config.openai_token, config.incorrect_models_dir,
                              seed_evaluation=True,
                              evaluation_path=os.path.join(config.seed_folder, "pre_evaluated_seed_metrics_custom.json")
                             )
    # Run the main evolutionary loop
    evo_prompt.evolve()
    evo_prompt.save_results(config.save_directory)
