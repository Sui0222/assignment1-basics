
from cs336_basics.bpe import train_bpe_optimized, save_tokenizer

vocab, merges = train_bpe_optimized(
    "data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)

save_tokenizer(vocab, merges, "data/tinystories_vocab.pkl", "data/tinystories_merges.pkl")

print("Tokenizer 訓練完成,已存檔。")
print("Vocab size:", len(vocab))