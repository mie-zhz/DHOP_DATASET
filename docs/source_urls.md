# Original Dataset Sources

This document lists the authoritative source of each underlying dataset and the expected local layout that `scripts/prepare_seeded_splits.py` consumes.

## Authoritative Sources

| Dataset | Original Paper / Source | Primary Distribution |
|---------|-------------------------|----------------------|
| BC5CDR | Li et al. (2016), BioCreative V CDR task | HuggingFace: <https://huggingface.co/datasets/bigbio/bc5cdr> |
| CoNLL-2003 | Tjong Kim Sang and De Meulder (2003), CoNLL-2003 shared task | HuggingFace: <https://huggingface.co/datasets/eriktks/conll2003> |
| CrossNER | Liu et al. (2020) | GitHub: <https://github.com/zliucr/CrossNER> |
| Few-NERD | Ding et al. (2021) | HuggingFace: <https://huggingface.co/datasets/DFKI-SLT/few-nerd> |

The CrossNER GitHub repository distributes all five domains used in this work (AI, Literature, Music, Politics, Science). The HuggingFace `DFKI-SLT/few-nerd` mirror corresponds to the supervised variant of Few-NERD.

## Expected Local Layout

The reproduction script reads each dataset using `datasets.load_from_disk`, which expects datasets already saved on local disk in the HuggingFace `DatasetDict` arrow format. Create the following directory under your working directory:

```
raw_dataset/
├── bc5cdr/                              # bigbio loader, then save_to_disk
├── conll2003/                           # eriktks/conll2003, then save_to_disk
├── few-nerd/                            # DFKI-SLT/few-nerd, then save_to_disk
└── cross_ner/
    ├── ai/
    ├── literature/
    ├── music/
    ├── politics/
    └── science/
```

Each leaf directory must be a HuggingFace `DatasetDict` containing `train`, `validation`, and `test` splits.

## Download Snippets

The exact loading commands below are the steps used to populate `raw_dataset/` in our internal pipeline. Adapt the dataset identifiers to your HuggingFace cache configuration as needed.

### BC5CDR

```python
from datasets import load_dataset
ds = load_dataset("bigbio/bc5cdr", name="bc5cdr_source", trust_remote_code=True)
ds.save_to_disk("raw_dataset/bc5cdr")
```

### CoNLL-2003

```python
from datasets import load_dataset
ds = load_dataset("eriktks/conll2003")
ds.save_to_disk("raw_dataset/conll2003")
```

### Few-NERD

```python
from datasets import load_dataset
ds = load_dataset("DFKI-SLT/few-nerd", name="supervised")
ds.save_to_disk("raw_dataset/few-nerd")
```

### CrossNER

CrossNER is distributed as raw CoNLL files on GitHub. Convert them to HuggingFace `DatasetDict` format before saving:

```bash
git clone https://github.com/zliucr/CrossNER.git /tmp/CrossNER
```

Then for each domain (replace `<domain>` with `ai`, `literature`, `music`, `politics`, `science`):

```python
from datasets import Dataset, DatasetDict

def read_conll(path):
    tokens, labels = [], []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    records.append({"tokens": tokens, "ner_tags": labels})
                    tokens, labels = [], []
                continue
            tok, tag = line.split("\t") if "\t" in line else line.split()
            tokens.append(tok)
            labels.append(tag)
    if tokens:
        records.append({"tokens": tokens, "ner_tags": labels})
    return records

base = "/tmp/CrossNER/data/"
domain = "ai"   # or literature/music/politics/science
ds = DatasetDict({
    split: Dataset.from_list(read_conll(f"{base}{domain}/{split}.txt"))
    for split in ("train", "validation", "test")
})
ds.save_to_disk(f"raw_dataset/cross_ner/{domain}")
```

The CrossNER repository uses `train.txt`, `valid.txt`, `test.txt` as split filenames; rename `valid.txt` to `validation.txt` if you prefer explicit split names.

## Verification

After populating `raw_dataset/`, run:

```bash
python scripts/prepare_seeded_splits.py \
    --raw-root raw_dataset \
    --output-root sampled_dataset \
    --seeds 2026 2027 2028 2029 2030
```

The generated splits should match `data/seeded_splits/` byte-for-byte modulo absolute paths stored in `sample_manifest.json` (the manifest records where the source data was located on disk at sampling time; sample indices, ordering, and contents are deterministic).

## Licensing Notes

Each underlying dataset retains its original license. Users of the derived splits in this repository must comply with the terms of every upstream dataset. See `LICENSE` for the license applied to the splits and scripts in this repository.
