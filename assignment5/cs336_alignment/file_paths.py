from pathlib import Path
import os

data_directory = Path('data')
model_direcotry = data_directory / 'models'
fine_tuned_model_directory = model_direcotry / 'fine_tuned'
datasets_direcotry = data_directory / 'datasets'
logs_directory = data_directory / 'logs'
os.makedirs(data_directory, exist_ok=True)
os.makedirs(model_direcotry, exist_ok=True)
os.makedirs(datasets_direcotry, exist_ok=True)
os.makedirs(logs_directory, exist_ok=True)

GSM8K_dataset = datasets_direcotry / 'GSM8K/main'
GSM8K_train = GSM8K_dataset / 'train-00000-of-00001.parquet'
GSM8K_test = GSM8K_dataset / 'test-00000-of-00001.parquet'

qwen_model_path = model_direcotry / 'Qwen2.5-Math-1.5B'
qwen_model_fine_tuned_path = fine_tuned_model_directory / 'Fine-Tuned-Qwen2.5-Math-1.5B'
qwen_math = "Qwen/Qwen2.5-Math-1.5B" 