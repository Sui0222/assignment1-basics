import os
from typing import BinaryIO

import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(file: BinaryIO,desired_num_chunks: int,split_special_token: bytes,) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    # 第一步:把每個 special token 都用 re.escape 處理過,存進一個新 list
    escaped_tokens = []
    for token in special_tokens:
        escaped_tokens.append(re.escape(token))  # TODO: 對 token 呼叫 re.escape

    # 第二步:把 escaped_tokens 這個 list,用 "|" 串成一個字串
    pattern = "|".join(escaped_tokens)

    # 第三步:用這個 pattern,呼叫 re.split,把 text 切開
    segments = re.split(pattern, text)

    return segments

def pretokenize_chunk(text: str) -> dict[tuple[bytes, ...], int]:
    """
    將一段文字做 pre-tokenization,回傳 pre-token 計數字典。
    例如輸入 "the cat and the dog",
    回傳類似 {(b't',b'h',b'e'): 2, (b' ',b'c',b'a',b't'): 1, ...}
    """
    counts: dict[tuple[bytes, ...], int] = {}

    # 第一步:用 PAT 這個 regex,把 text 切成一堆字串片段(pre-tokens)
    # 提示:用 re.findall(PAT, text)
    pieces = re.findall(PAT,text)  # TODO: 填入正確的呼叫

    # 第二步:對每個字串片段(piece),做以下事情:
    for piece in pieces:
        # 2a. 把這個字串片段轉成 utf-8 bytes
        piece_bytes = piece.encode("utf-8")  # TODO

        # 2b. 把這個 bytes 物件,拆成「每個元素是長度1的bytes」的 tuple
        #     注意:直接 for b in piece_bytes 會拿到 int,不是 bytes
        #     你需要把每個 int 包回 bytes(提示: bytes([b]))
        symbol_tuple = tuple(bytes([b]) for b in piece_bytes)  

        # 2c. 把這個 symbol_tuple 累加進 counts 字典
        #     提示: dict 的 .get(key, 0) 方法可以幫你處理「key 不存在時預設為 0」的情況
        if symbol_tuple in counts:
            counts[symbol_tuple] = counts[symbol_tuple] + 1
        else:
            counts[symbol_tuple] = 1

    return counts

def merge_counts(dict_a: dict[tuple[bytes, ...], int],dict_b: dict[tuple[bytes, ...], int],) -> dict[tuple[bytes, ...], int]:
    """
    把 dict_b 的內容合併進 dict_a(相同的 key,數值要相加),
    回傳合併後的結果。
    """
    merged = dict(dict_a)  # 先複製一份 dict_a,避免直接修改到原本的 dict_a

    for key in dict_b:
        # TODO: 判斷 key 是否已經在 merged 裡
        #   如果在 → merged[key] 要加上 dict_b[key]
        #   如果不在 → merged[key] 直接設成 dict_b[key]
        if merged.get(key)!=None:
            merged[key]=merged[key]+dict_b[key]
        else:
            merged[key]=dict_b[key]

    return merged

def pretokenize_text(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:

    segments = split_on_special_tokens(text, special_tokens)
    total_counts: dict[tuple[bytes, ...], int] = {}

    for segment in segments:
        # TODO 1: 對這個 segment 呼叫 pretokenize_chunk,拿到這一段的計數字典
        segment_counts = pretokenize_chunk(segment)
        # TODO 2: 把 segment_counts 合併進 total_counts
        total_counts=merge_counts(total_counts,segment_counts)

    return total_counts

text = "the cat and the cat<|endoftext|>the dog and the cat"
print(pretokenize_text(text, ["<|endoftext|>"]))