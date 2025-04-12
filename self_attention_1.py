import torch
inputs = torch.tensor([
    [0.43, 0.15, 0.89], 
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64], 
    [0.22, 0.58, 0.33], 
    [0.77, 0.25, 0.10], 
    [0.05, 0.80, 0.55]
])

query = inputs[1]
attn_score_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_score_2[i] = torch.dot(x_i, query)

print(attn_score_2)

# normalization
attn_weights_2_tmp = attn_score_2 / attn_score_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# normalization using softmax - output can be interpreted as probabilities
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_score_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# softmax from torch (other one may raise numerical instability problems)
attn_weights_2 = torch.softmax(attn_score_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# context vector - weighted sum
# what makes this the context vector 2 is multiplying by query 2 at previous step for weigths
query = inputs[1]
context_vec_2 = torch.empty(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)

# compute weights for each input as query

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)