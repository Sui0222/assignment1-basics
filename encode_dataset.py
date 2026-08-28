import numpy as np
from cs336_basics.tokenizer import Tokenizer

def encode_file_to_npy(tokenizer, input_path, output_path):
    ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)

    ids_array = np.array(ids, dtype=np.uint16)
    np.save(output_path, ids_array)
    print(f"{input_path} -> {output_path}, 共 {len(ids_array)} 個 token")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "data/tinystories_vocab.pkl",
        "data/tinystories_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    encode_file_to_npy(tokenizer, "data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_train.npy")
    encode_file_to_npy(tokenizer, "data/TinyStoriesV2-GPT4-valid.txt", "data/tinystories_valid.npy")