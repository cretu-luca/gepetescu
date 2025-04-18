import tiktoken
import json
import torch
from model import GPT
from tokenization_utils import text_to_token_ids, token_ids_to_text
from generate import generate

with open('config.json', 'r') as f:
    config = json.load(f)

def main():

    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT(config)

    checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Say hi to", tokenizer),
        max_new_tokens=15,
        context_size=config["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == '__main__':
    main()