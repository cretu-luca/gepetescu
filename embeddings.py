import torch
from data_loader import create_dataloader_v1

# preparation involves tokenizing text, converting text tokens to token ID's and 
# converting token ID's into embedding vectors
vocab_size = 50257 # words
output_dim = 256 # dimension of one embedding

# torch.nn.Embedding(num_embeddings, embedding_dim, ...)
# num_embeddings (int) – size of the dictionary of embeddings
# embedding_dim (int) – the size of each embedding vector

torch.manual_seed(123)
token_embeddings_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embeddings_layer.weight)

# torch.nn.Embedding -> look-up
# print(embeddings_layer(torch.tensor([3])))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embeddings_layer(inputs)
print(token_embeddings.shape)

print(token_embeddings)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)