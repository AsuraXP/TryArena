"""build_corpus.py — assemble a ~100 MB (configurable) plain-English prose corpus from GitHub-hosted public-domain books.

Source: nschaetti/SFGram-dataset (public-domain science-fiction novels and pulp magazines, Project Gutenberg texts;
1390 files, 267 MB).  Fetched with a sparse git clone (github.com only), so it works both on the Actions runners
and anywhere else.  Nothing is stored in this repo: the corpus is rebuilt deterministically (same cleaning, same
book order) wherever it is needed.

Cleaning: strip Gutenberg header/footer, drop italics underscores, unwrap hard line breaks inside paragraphs,
collapse whitespace, keep ASCII only (byte-level models), skip files < 20 kB (tables of contents / fragments).
Books are concatenated in file order; the first MB bytes are written.  Held-out split is the job of the trainer
(last 5 % of the file, as in arch_vet_p49.lm_run).

usage: python build_corpus.py --out corpus/prose100.txt --mb 100
"""
import argparse, glob, os, re, subprocess

def clean(t):
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t, re.S | re.I)
    if m: t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t, re.I)
    if m: t = t[:m.start()]
    t = t.replace("\r", "").replace("_", " ")
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t); t = re.sub(r"[ ]{2,}", " ", t)
    t = "".join(ch for ch in t if ord(ch) < 128)
    return t.strip() + "\n\n"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="corpus/prose100.txt"); ap.add_argument("--mb", type=float, default=100)
    ap.add_argument("--src", default="/tmp/sfgram"); a = ap.parse_args()
    if not os.path.isdir(os.path.join(a.src, "book-contents")):
        subprocess.check_call(["git", "clone", "-q", "--filter=blob:none", "--sparse", "--depth", "1", "https://github.com/nschaetti/SFGram-dataset", a.src])
        subprocess.check_call(["git", "-C", a.src, "sparse-checkout", "set", "book-contents"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); lim = int(a.mb * 1e6); tot = n = 0
    with open(a.out, "w") as out:
        for f in sorted(glob.glob(os.path.join(a.src, "book-contents", "*.txt"))):
            t = open(f, errors="ignore").read()
            if len(t) < 20000: continue
            c = clean(t)[: max(0, lim - tot)]; out.write(c); tot += len(c); n += 1
            if tot >= lim: break
    print(f"wrote {a.out}: {n} books, {tot/1e6:.1f} MB")

if __name__ == "__main__": main()
