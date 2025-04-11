with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print("Total number of characters: ", len(raw_text))
# print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
# print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        # inverse vocabulary that maps token ID's back to the original text tokens
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        # processes input text into token ID's
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        # converts token ID's back into text
        text = " ".join([self.int_to_str[i] for i in ids])

        # removes spaces before the specified punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
# tokenizing a passage
tokenizer = SimpleTokenizerV1(vocab=vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)

# the token ID's obtained by encoding
print("PRINTING IDs")
print(ids)

# the words obtained by decoding
print(tokenizer.decode(ids))

# addings special tokens for unknown words (<|unk|>) and end of source (<|endoftext|>)
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# initial text
print(tokenizer.encode(text))