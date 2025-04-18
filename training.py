import json
import torch
import torch.nn as nn
import tiktoken
import argparse

from data_loader import create_dataloader_v1
from tokenization_utils import text_to_token_ids, token_ids_to_text
from generate import generate, generate_and_print_sample
from model import GPT

file_path = "ad/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

with open('config.json', 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", nargs='?')

# cross entropy loss of a given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss

# compute loss over all batches sampled by a given data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, 
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"ep {epoch+1} (step {global_step:06d}): "
                    f"train loss {train_loss:.3f}, "
                    f"val loss {val_loss:.3f}"
                )
    '''
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    '''

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )

        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
            lr=0.0004, weight_decay=0.1)

    args = parser.parse_args()
    if args.filename is not None:
        print(f"loading model from {args.filename}")

        checkpoint = torch.load(args.filename, weights_only=True) 

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:
        print(f"initializing untrained model")

    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=1, eval_iter=5,
        start_context="every effort moves you", tokenizer=tokenizer
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"
    )

if __name__ == '__main__':
    main()