from .pretokenization import PAT   # encode 時應該也需要同樣的 regex 切詞規則
import regex as re
from .bpe import merge_pair_in_symbols
import pickle

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_reverse = {}
        self.merge_priority={}
        for token_id, token_bytes in vocab.items():
            # TODO: 把這一組資料,反過來存進 self.vocab_reverse
            self.vocab_reverse[token_bytes] = token_id

        for index, pair in enumerate(self.merges):
            self.merge_priority[pair]=index
            

    # tokenizer.py 的 from_file
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        special_tokens = self.special_tokens if self.special_tokens else []
        segments = split_keep_special_tokens(text, special_tokens)

        ids = []
        for segment in segments:
            if segment in special_tokens:
                # TODO ①: 這是 special token,直接查 self.vocab_reverse 拿 id,加進 ids
                ids.append(self.vocab_reverse.get(segment.encode("utf-8")))
            else:
                # 一般文字,先用 PAT 做 pre-tokenize
                pieces = re.findall(PAT, segment)
                for piece in pieces:
                    piece_bytes = piece.encode("utf-8")
                    symbols = tuple(bytes([b]) for b in piece_bytes)

                    # TODO ②: 呼叫 encode_symbols,套用合併規則
                    merged_symbols =encode_symbols(symbols,self.merge_priority)

                    # TODO ③: 把 merged_symbols 裡每一個 symbol,查 vocab_reverse 轉成 id,加進 ids
                    for symbol in merged_symbols:
                        id=self.vocab_reverse.get(symbol)
                        ids.append(id)
                    

        return ids

    def encode_iterable(self, iterable):
        for text_chunk in iterable:
            #逐行讀取S每一段文字
            ids=self.encode(text_chunk)
            for token_id in ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:# ids 待轉換的list
        all_bytes = b""   # 空的 bytes 物件,當作累加的起點
        for token_id in ids:
        # TODO: 從 self.vocab 查出這個 id 對應的 bytes,累加進 all_bytes
            all_bytes = all_bytes + self.vocab[token_id]
        # TODO: 把累加完的 all_bytes,一次性 decode 成文字,回傳
        return all_bytes.decode("utf-8",errors="replace")

def get_best_pair_to_merge(symbols, merge_priority):
    # 第一步:找出 symbols 裡目前所有的相鄰 pair(這你已經很熟悉了,
    #         跟 count_pairs 內層迴圈邏輯一樣)
    candidates = []
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
        # TODO: 只有「這個 pair 存在於 merge_priority 裡」時,才加進 candidates
        if merge_priority.get(pair)==None:continue
        candidates.append(pair)

    # 第二步:如果 candidates 是空的,代表沒有任何可以合併的了
    if not candidates:
        return None

    # 第三步:從 candidates 裡,選出 merge_priority 最小的那一個
    best = min(candidates, key=lambda pair: merge_priority[pair])
    return best
def encode_symbols(symbols, merge_priority):
    """
    對單一個 pre-token(symbol tuple),重複套用合併規則,直到沒有可合併的為止。
    """
    while True:
        best_pair = get_best_pair_to_merge(symbols, merge_priority)
        if best_pair is None:
            break
        symbols = merge_pair_in_symbols(symbols, best_pair)
    return symbols

def split_keep_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    依照 special tokens 切開文字,但保留 special token 本身在結果列表裡。
    例如 split_keep_special_tokens("hello<|endoftext|>world", ["<|endoftext|>"])
    → ['hello', '<|endoftext|>', 'world']
    """
    if not special_tokens:
        return [text]

    escaped_tokens = []
    for token in sorted(special_tokens, key=len, reverse=True):
        escaped_tokens.append(re.escape(token))
    pattern = "(" + "|".join(escaped_tokens) + ")"   # 注意:多包了一層括號

    segments = re.split(pattern, text)
    return segments

if __name__ == "__main__":
    from .bpe import train_bpe, save_tokenizer

    vocab, merges = train_bpe("/tmp/toy.txt", vocab_size=260, special_tokens=["<|endoftext|>"])
    save_tokenizer(vocab, merges, "/tmp/vocab.pkl", "/tmp/merges.pkl")
    tokenizer = Tokenizer.from_files("/tmp/vocab.pkl", "/tmp/merges.pkl", special_tokens=["<|endoftext|>"])

    text = "the cat and the cat<|endoftext|>the dog"
    ids = tokenizer.encode(text)
    print("Encoded ids:", ids)

    decoded_text = tokenizer.decode(ids)
    print("Decoded text:", decoded_text)
    print("Round-trip 成功嗎?", decoded_text == text)

    # 測試 encode_iterable:模擬逐行讀取一個檔案
    lines = ["the cat and the cat\n", "the dog\n"]
    ids_from_iterable = list(tokenizer.encode_iterable(lines))
    print("encode_iterable ids:", ids_from_iterable)

    # 拿同樣的文字內容,直接用 encode 一次做,結果應該要一致
    ids_from_encode = tokenizer.encode("the cat and the cat\nthe dog\n")
    print("直接 encode ids:", ids_from_encode)
    print("兩者相同嗎?", ids_from_iterable == ids_from_encode)