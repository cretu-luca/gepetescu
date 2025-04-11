from importlib.metadata import version
import tiktoken

# print(version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

strings = tokenizer.decode(integers)
# print(strings)

unknown = "Akwirw ier"
ids = tokenizer.encode(unknown, allowed_special={"<|endoftext|>"})
print(ids)

strings2 = tokenizer.decode(ids)
print(strings2)
