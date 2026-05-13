import torch
import gradio as gr

from filepath import *
from cs336_basics.inference import generate_text
from cs336_basics.tokenizer import tokenizer

TS_tokenizer = tokenizer.from_files(vocab_filepath=TS_VOCAB_PATH, merges_filepath=TS_MERGES_PATH, special_tokens=SPECIAL_TOKENS)
TS_model = torch.load(TS_MODEL_PATH, weights_only=False)

def ts_lm_interface(prompt, temperature, top_p, context_length):
    output = generate_text(
        model=TS_model,
        tokenizer=TS_tokenizer,
        prompt=prompt,
        context_length=int(context_length),
        temperature=float(temperature),
        top_p=float(top_p)
    )
    return output

ts_demo = gr.Interface(
    fn=ts_lm_interface,
    inputs=[
        gr.Textbox(label="Prompt", lines=2, placeholder="Type your prompt here..."),
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature"),
        gr.Slider(0.5, 1.0, value=0.85, label="Top-p"),
        gr.Slider(32, 1024, value=256, step=32, label="Context Length")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="TinyStory Language Model",
    description="Enter a prompt and adjust the decoding parameters to generate text with your trained language model.",
    theme="soft"
)

if __name__ == "__main__":
    ts_demo.launch()