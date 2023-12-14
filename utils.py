from dataclasses import dataclass
import yaml


@dataclass
class Config:
    openai_token: str
    train_dataset_path: str
    test_dataset_path: str
    target_name: str
    save_directory: str
    seed_folder: str
    incorrect_models_dir: str
    rounds_count: int = 10
    few_shot_prompts_per_round: int = 3
    samples_per_prompt: int = 3
    in_context_examples_per_prompt: int = 2
    survivors_per_generation: int = 5
    n_evaluations: int = 3
    alpha: int = 600000
    dataset_type: str = "classification"
    openai_model_name: str = "gpt-4-1106-preview"
    target_model_factor: float = 0.90 
    target_metric: float = 0.95
    batch_size: int = 1000
    device: str = "cuda:0"


def load_config(path: str):
    with open(path, 'r') as file:
        loaded_config = yaml.safe_load(file)
    config = Config(**loaded_config)
    return config
