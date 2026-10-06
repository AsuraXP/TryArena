"""
Byte-level BPE tokenizer, from scratch (C21 fluency-axis tooling).
Vocab = 256 bytes + 1 space token + merge rules. Fits on a corpus subset,
caches to corpus/tok_cache.pkl. Usage:
  python3 bpe_tok.py            # fit + cache + report
  from bpe_tok import load_tk   # encode/decode helpers
"""
import math, os, pickle, re, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus", "corpus_full.txt")
CACHE = os.path.join(HERE, "corpus", "tok_cache.pkl")
SPACE = 256                      # token id for whitespace
TARGET_VOCAB = 768
MAX_WORD = 40                    # skip pathological long byte-words in fitting


def read_corpus():
    with open(CORPUS, "rb") as f:
        return f.read()


def split_words(text: bytes):
    return [w for w in re.split(rb"[\n\t\r ]+", text) if w and len(w) <= MAX_WORD]


def fit_bpe(text: bytes, seed=0):
    words = split_words(text)
    print(f"[bpe] fitting on {len(words)} words ({len(text)} bytes)", flush=True)
    word_tokens = [list(w) for w in words]
    merges = []
    next_id = 257
    t0 = time.time()
    while next_id < TARGET_VOCAB:
        pairs = Counter()
        for toks in word_tokens:
            for a, b in zip(toks, toks[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        merges.append((a, b, next_id))
        new = next_id
        next_id += 1
        nxt = []
        for toks in word_tokens:
            if a not in toks:          # cheap skip
                nxt.append(toks); continue
            out, i = [], 0
            while i < len(toks):
                if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                    out.append(new); i += 2
                else:
                    out.append(toks[i]); i += 1
            nxt.append(out)
        word_tokens = nxt
        if len(merges) % 100 == 0:
            print(f"[bpe] merges {len(merges)}/{TARGET_VOCAB - 257} ({time.time()-t0:.0f}s)", flush=True)
    return merges


def build(merges):
    token_bytes = {i: bytes([i]) for i in range(256)}
    token_bytes[SPACE] = b" "
    pair_to_new = {}
    for a, b, n in merges:
        token_bytes[n] = token_bytes[a] + token_bytes[b]
        pair_to_new[(a, b)] = n
    vocab_size = 257 + len(merges)

    def encode_word(w: bytes):
        toks = list(w)
        while True:
            hit = None
            for i in range(len(toks) - 1):
                p = pair_to_new.get((toks[i], toks[i + 1]))
                if p is not None:
                    if hit is None or p < hit[0]:      # lowest merge-order wins
                        hit = (p, i)
            if hit is None:
                break
            p, i = hit
            toks = toks[:i] + [p] + toks[i + 2:]
        return toks

    def encode(text: bytes):
        ids = []
        for w in re.split(rb"[\n\t\r ]+", text):
            if not w:
                continue
            if ids:
                ids.append(SPACE)
            if len(w) > MAX_WORD:                     # truncate, keep bytes (coverage)
                ids.extend(list(w[:MAX_WORD]))
            else:
                ids.extend(encode_word(w))
        return ids

    def decode(ids):
        return b"".join(token_bytes[i] for i in ids)

    return vocab_size, encode, decode


def main():
    text = read_corpus()
    # fit on a 300KB subset (speed); the full corpus is used for training
    subset = text[:300_000]
    merges = fit_bpe(subset)
    vocab_size, encode, decode = build(merges)
    print(f"[bpe] vocab {vocab_size}, merges {len(merges)}", flush=True)
    ids = encode(text)
    print(f"[bpe] corpus tokens: {len(ids)} (full {len(text)} bytes)", flush=True)
    sample = encode(b"It is a truth universally acknowledged that a single man in possession of a good fortune")
    print("[bpe] sample:", [decode([i]) for i in sample][:24])
    with open(CACHE, "wb") as f:
        pickle.dump({"merges": merges, "vocab_size": vocab_size}, f)
    print("[bpe] cached ->", CACHE, flush=True)


def load_tk():
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    assert d["vocab_size"] == 257 + len(d["merges"])
    return build(d["merges"])                       # (vocab_size, encode, decode)


if __name__ == "__main__":
    main()
