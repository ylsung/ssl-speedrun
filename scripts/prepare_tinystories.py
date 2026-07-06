"""Download TinyStories, train a small BPE tokenizer, encode to uint16 memmaps.
Run on the GPU instance (dataset lives there, not on the laptop):

    python scripts/prepare_tinystories.py --out data/tinystories --vocab_size 4096

Writes: tokenizer.json, meta.json, train.bin, val.bin.
Needs: pip install datasets tokenizers  (see requirements-data.txt)
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--tokenizer_stories", type=int, default=200_000,
                    help="stories used to train the BPE")
    ap.add_argument("--max_stories", type=int, default=None,
                    help="cap stories encoded (default: all)")
    args = ap.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")

    tok_path = os.path.join(args.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
        print(f"reusing {tok_path}")
    else:
        print(f"training BPE vocab={args.vocab_size} "
              f"on {args.tokenizer_stories} stories...")
        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        it = (ds["train"][i]["text"] for i in range(
            min(args.tokenizer_stories, len(ds["train"]))))
        tokenizer.train_from_iterator(it, trainer=trainer)
        tokenizer.save(tok_path)

    eot = tokenizer.token_to_id("<|endoftext|>")
    assert eot is not None

    def encode_split(split, path, cap=None):
        n = len(ds[split]) if cap is None else min(cap, len(ds[split]))
        chunks, total = [], 0
        B = 10_000
        for lo in range(0, n, B):
            texts = ds[split][lo:lo + B]["text"]
            for enc in tokenizer.encode_batch(texts):
                ids = enc.ids + [eot]
                chunks.append(np.asarray(ids, dtype=np.uint16))
                total += len(ids)
            print(f"\r{split}: {min(lo + B, n)}/{n} stories, {total/1e6:.1f}M tok",
                  end="", flush=True)
        arr = np.concatenate(chunks)
        arr.tofile(path)
        print(f"\n{path}: {len(arr)/1e6:.1f}M tokens")
        return len(arr)

    n_train = encode_split("train", os.path.join(args.out, "train.bin"),
                           args.max_stories)
    n_val = encode_split("validation", os.path.join(args.out, "val.bin"))

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"vocab_size": tokenizer.get_vocab_size(),
                   "eot_id": eot, "train_tokens": n_train,
                   "val_tokens": n_val, "dataset": "roneneldan/TinyStories"},
                  f, indent=2)
    print("done:", os.path.join(args.out, "meta.json"))


if __name__ == "__main__":
    main()
