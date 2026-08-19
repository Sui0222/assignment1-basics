from . import pretokenization

def count_pairs(counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Given pre-token counts, return the weighted frequency of every adjacent symbol pair.
    """
    pair_counts: dict[tuple[bytes, bytes], int] = {}#初始化

    for symbols, count in counts.items(): # symbols:=key,count:=value
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            # TODO: add `count` to pair_counts[pair] (create if not present)
            if  pair_counts.get(pair) != None:
                pair_counts[pair]=pair_counts[pair]+count
            else:
                pair_counts[pair]=count

    return pair_counts


def merge_pair_in_symbols(symbols: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    #合并操作
    new_symbols = []
    i = 0
    while i<len(symbols):
        # 檢查:目前位置(i)開始的兩個 symbol,是不是等於 pair?
        # 注意:i+1 不能超出範圍(最後一個 symbol 沒有「下一個」可比)
        if i<len(symbols)-1 and (symbols[i],symbols[i+1])==pair:
            new_symbols.append(symbols[i]+symbols[i+1])
            i=i+2
        else:
            # 沒找到,這個 symbol 原封不動保留
            new_symbols.append(symbols[i])
            i=i+1

    return tuple(new_symbols)

def apply_merge(counts: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    """
    對整個 counts 字典的每一個 pre-token,套用合併操作,回傳新的計數字典。
    """
    new_counts: dict[tuple[bytes, ...], int] = {}

    for symbols, count in counts.items():
        new_symbols = merge_pair_in_symbols(symbols, pair)
        # TODO: 把 new_symbols 這個 key,累加 count 進 new_counts
        #       (提示:這個邏輯你已經寫過兩次了——一次在 pretokenize_chunk 的 2c,
        #        一次在 merge_counts 裡)
        if new_counts.get(new_symbols)!=None:
            new_counts[new_symbols]=new_counts[new_symbols]+count
        else:
            new_counts[new_symbols]=count
    return new_counts

def train_bpe(input_path: str,vocab_size: int,special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 第一步:讀取檔案內容
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 第二步:取得初始 pre-token 計數
    counts = pretokenization.pretokenize_text(text, special_tokens)

    # 第三步:初始化 vocab(256 個 byte + special tokens)
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id = next_id + 1

    # 第四步:主迴圈
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # TODO 1: 呼叫 count_pairs,取得目前的 pair 頻率
        pair_counts = count_pairs(counts)
        # TODO 2: 如果 pair_counts 是空的(代表沒有可以合併的了),要跳出迴圈
        #         (提示:用 if not pair_counts: break)
        if not pair_counts:break
        # TODO 3: 選出最高頻的 pair(用我們之前確認過的 max(...) 那行)
        best_pair = max(pair_counts,key=lambda pair: (pair_counts[pair], pair))
        # TODO 4: 呼叫 apply_merge,更新 counts
        counts = apply_merge(counts,best_pair)#合并best_pair
        # TODO 5: 把 best_pair 加進 merges 列表
        merges.append(best_pair)
        # TODO 6: 把合併後的新 best_pair[0] + best_pair[1] 加進 vocab,分配 next_id,並遞增 next_id
        vocab[next_id] = best_pair[0]+best_pair[1]
        next_id=next_id+1

    return vocab, merges

def build_pair_index(counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    """
    對每個 pre-token,找出它包含的所有 pair,
    建立一份「pair -> 有哪些 pre-token 含有它」的反查表。
    """
    index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for symbols in counts:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            # TODO: 把 symbols 這個 pre-token,加進 index[pair] 這個 set 裡
            #       (如果 index[pair] 還不存在,要先建立一個空的 set)
            if index.get(pair)==None:
                index[pair]={symbols}
            else :
                index[pair].add(symbols)

    return index

def train_bpe_optimized(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    counts = pretokenization.pretokenize_text(text, special_tokens)

    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id = next_id + 1

    merges: list[tuple[bytes, bytes]] = []

    # 只算「一次」初始的 pair_counts 和 index
    pair_counts = count_pairs(counts)
    pair_index = build_pair_index(counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        # 接下來要寫:只更新受影響的 pre-token
        # TODO: 呼叫一個新函式,做「增量更新」這件事
        counts, pair_counts, pair_index = apply_merge_incremental(counts, pair_counts, pair_index, best_pair)

        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id = next_id + 1

    return vocab, merges

def apply_merge_incremental(
    counts: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    best_pair: tuple[bytes, bytes],
) -> tuple[dict, dict, dict]:

    # 找出所有受影響的 pre-token(注意:要先複製一份,因為等一下會修改 pair_index,
    # 如果直接迭代原本的 set 同時又修改它,會出錯)
    affected = list(pair_index[best_pair])

    for old_symbols in affected:
        count = counts[old_symbols]

        # 第一步:扣除 old_symbols 對 pair_counts 的舊貢獻
        # TODO ①
        for i in range(len(old_symbols) - 1):
            p = (old_symbols[i], old_symbols[i + 1])
            pair_counts[p] = pair_counts[p] - count
            if pair_counts[p] <= 0: #已經清空的就不該存在了
                del pair_counts[p]

        # 第二步:算出合併後的新 symbol tuple
        new_symbols = merge_pair_in_symbols(old_symbols, best_pair)

        # 第三步:加上 new_symbols 對 pair_counts 的新貢獻
        for i in range(len(new_symbols) - 1):
            p = (new_symbols[i], new_symbols[i + 1])
            if pair_counts.get(p) != None:
                pair_counts[p] = pair_counts[p] + count
            else:
                pair_counts[p] = count

        # 第四步:# 移除 old_symbols(它已經不是目前的狀態了)
        del counts[old_symbols]

        # 加上或累加 new_symbols
        if counts.get(new_symbols) != None:
            counts[new_symbols] = counts[new_symbols] + count
        else:
            counts[new_symbols] = count

        # 第五步:更新 pair_index(把 old_symbols 從它涉及的 pair 移除,
        #         把 new_symbols 加進它涉及的 pair)# A. 把 old_symbols 從它舊涉及的 pair 中移除
        for i in range(len(old_symbols) - 1):
            p = (old_symbols[i], old_symbols[i + 1])
            # TODO: 把 old_symbols 從 pair_index[p] 這個 set 中移除
            pair_index[p].discard(old_symbols)

        # B. 把 new_symbols 加進它新涉及的 pair
        for i in range(len(new_symbols) - 1):
            p = (new_symbols[i], new_symbols[i + 1])
            # TODO: 把 new_symbols 加進 pair_index[p](如果 p 還不存在,先建立空 set)
            if pair_index.get(p)==None:
                pair_index[p]={new_symbols}
            else:
                pair_index[p].add(new_symbols)

    return counts, pair_counts, pair_index

import pickle

def save_tokenizer(vocab, merges, vocab_path, merges_path):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)