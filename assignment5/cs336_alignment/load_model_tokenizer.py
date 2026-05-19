import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

try:
    import file_paths
except:
    try:
        from cs336_alignment import file_paths
    except:
        ImportError

local_model_path = file_paths.qwen_model_path
model_name = file_paths.qwen_math

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def local_model_ready(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        and (path / "tokenizer.json").exists()
        and (path / "tokenizer_config.json").exists()
        and (
            (path / "model.safetensors").exists()
            or (path / "model.safetensors.index.json").exists()
        )
    )

def load_model_tokenizer(model_name=model_name, local_model_path:Path =local_model_path, dtype=dtype, device=device) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # if local_model_path.exists():
    if local_model_ready(local_model_path):
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True, fix_mistral_regex=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            local_files_only=True,
            dtype=dtype,
        )
        # print("Loaded from local disk.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

        local_model_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
        print("Downloaded and saved to local disk.")
    return model.to(device), tokenizer

policy, model = load_model_tokenizer()