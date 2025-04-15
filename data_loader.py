import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    # collection of data points - defines how data is accesed and organized
    # when creating a new dataset, i must implement at least two methods - __len__() and __getitem__(idx)
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, # drop last batch if incomplete
        num_workers=num_workers # number of CPU processes to use for preprocessing
    )

    return dataloader

# conclusions  
# max_length - how many ids in one tensor
# batch_size - how many tensors in one batch
# stride - how many positions the inputs shift across batches
# therefore, when stride=max_length, no overlap between batches

# batch_size - tradeoff and hyperparameter to experiment with