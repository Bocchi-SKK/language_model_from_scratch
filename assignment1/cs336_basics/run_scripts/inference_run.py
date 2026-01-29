import torch
import gradio as gr

from filepath import *
from cs336_basics.inference import generate_text
from cs336_basics.tokenizer import tokenizer

# # TS_model_path = Path('E:\\Data\\OneDrive\\Data\\Code\\Python\\Language_Modeling_From_Scratch\\data\\model\\TS_model.pt')
# OWT_model_path = Path('E:\\Data\\OneDrive\\Data\\Code\\Python\\Language_Modeling_From_Scratch\\data\\model\\OWT_model.pt')
# # TS_tokenizer = tokenizer.from_files(vocab_filepath=TS_VOCAB_PATH, merges_filepath=TS_MERGES_PATH, special_tokens=SPECIAL_TOKENS)
# OWT_tokenizer = tokenizer.from_files(vocab_filepath=OWT_VOCAB_PATH, merges_filepath=OWT_MERGES_PATH, special_tokens=SPECIAL_TOKENS)

# # TS_model = torch.load(TS_model_path, weights_only=False)
# OWT_model = torch.load(OWT_model_path, weights_only=False)
# prompt = "The discovery of a new exoplanet has astronomers wondering if"
# output = generate_text(model=OWT_model, tokenizer=OWT_tokenizer, prompt=prompt, context_length=1024, temperature=0.7, top_p=0.85)
# print('=='*20)
# print("The text your model generate:")
# print(output)

OWT_model_path = Path('E:\\Data\\OneDrive\\Data\\Code\\Python\\Language_Modeling_From_Scratch\\data\\model\\OWT_model.pt')
OWT_tokenizer = tokenizer.from_files(vocab_filepath=OWT_VOCAB_PATH, merges_filepath=OWT_MERGES_PATH, special_tokens=SPECIAL_TOKENS)
OWT_model = torch.load(OWT_model_path, weights_only=False)

def lm_interface(prompt, temperature, top_p, context_length):
    output = generate_text(
        model=OWT_model,
        tokenizer=OWT_tokenizer,
        prompt=prompt,
        context_length=int(context_length),
        temperature=float(temperature),
        top_p=float(top_p)
    )
    return output

demo = gr.Interface(
    fn=lm_interface,
    inputs=[
        gr.Textbox(label="Prompt", lines=2, placeholder="Type your prompt here..."),
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature"),
        gr.Slider(0.5, 1.0, value=0.85, label="Top-p"),
        gr.Slider(32, 1024, value=1024, step=32, label="Context Length")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="OpenWebText Language Model",
    description="Enter a prompt and adjust the decoding parameters to generate text with your trained language model.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()