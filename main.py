import tiktoken
import json
import torch
from model import GPT
from tokenization_utils import text_to_token_ids, token_ids_to_text

with open('config.json', 'r') as f:
    config = json.load(f)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def main():

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPT(config)

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=config["context_length"]
    )

    # print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    inputs = torch.tensor([[16833, 3626, 6100],
                           [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345 ], 
                            [1107, 588, 11311]])
    
    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)



if __name__ == '__main__':
    main()