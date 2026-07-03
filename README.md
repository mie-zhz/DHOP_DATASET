# DHOP\_DATASET

Seeded train--test splits and sampling scripts for **generative named entity recognition (NER)** prompt-optimization experiments.

This repository accompanies the paper:

> *From Local Feedback to Persistent Evidence: Observable Prompt Optimization for Generative Named Entity Recognition.*

It contains the **exact data splits** used in the main experiments and ablations, together with the deterministic script that reproduces them from the original public datasets.

## Contents

```
.
├── data/
│   ├── seeded_split_summary.json      # 40-row manifest: 5 seeds × 8 datasets
│   └── seeded_splits/
│       └── seed_{2026,2027,2028,2029,2030}/
│           ├── bc5cdr/
│           ├── conll2003/
│           ├── cross_ner/{ai,literature,music,politics,science}/
│           └── few-nerd/
├── scripts/
│   ├── prepare_seeded_splits.py        # entry-point for reproducing splits
│   └── common/dataset_sampling.py      # sampling + conversion logic
├── docs/
│   └── source_urls.md                  # original dataset sources
├── README.md
├── LICENSE                             # CC-BY-4.0 (for the splits and scripts)
└── CITATION.cff
```

For each (seed, dataset) pair, the split directory contains:

| File | Size | Role |
|------|------|------|
| `runtime_all.json`         | 200 | Full optimization set (used by PromptBreeder and DHOP). |
| `runtime_train.json`       | 100 | 2-way train split (OPRO, GEPA). |
| `runtime_dev.json`         | 100 | 2-way dev split (OPRO, GEPA). |
| `runtime_train_test_train.json` | 80 | 3-way train split (PromptAgent). |
| `runtime_train_test_test.json`  | 60 | 3-way dev split (PromptAgent). |
| `runtime_test.json`        | 60  | 3-way test split (PromptAgent). |
| `blind_test.json`          | 500 | Held-out test set, never visible during optimization. |
| `sample_manifest.json`     | --- | Per-sample raw index provenance (split + index in the original HuggingFace/GitHub dataset). |

All JSON files share a common record schema with `text` and `entities` fields, where each entity has `text`, `label`, `start`, and `end`.

## Datasets and Original Sources

| Dataset | Domain | #Types | Source |
|---------|--------|:------:|--------|
| BC5CDR | Biomedical | 2 | https://huggingface.co/datasets/bigbio/bc5cdr |
| CoNLL-2003 | News | 3 | https://huggingface.co/datasets/eriktks/conll2003 |
| CrossNER-AI | Artificial intelligence | 5 | https://github.com/zliucr/CrossNER |
| CrossNER-Literature | Literature | 5 | https://github.com/zliucr/CrossNER |
| CrossNER-Music | Music | 5 | https://github.com/zliucr/CrossNER |
| CrossNER-Politics | Politics | 5 | https://github.com/zliucr/CrossNER |
| CrossNER-Science | Science | 5 | https://github.com/zliucr/CrossNER |
| Few-NERD | General fine-grained | 5 | https://huggingface.co/datasets/DFKI-SLT/few-nerd |

Detailed source URLs and download instructions are listed in [`docs/source_urls.md`](docs/source_urls.md).

## Reproducing the Splits

The splits are produced deterministically from the original datasets using the five fixed random seeds `{2026, 2027, 2028, 2029, 2030}`. For each (seed, dataset) pair, 700 instances are sampled without replacement: 200 form the optimization set and 500 form the held-out test set. Method-specific sub-splits are then derived from the 200 optimization instances.

Prerequisites:

- Python ≥ 3.10
- The `datasets` library (HuggingFace): `pip install datasets`

Steps:

```bash
# 1. Download the original datasets (see docs/source_urls.md for exact commands).
#    Place each dataset on local disk under raw_dataset/, for example:
#
#      raw_dataset/
#      ├── bc5cdr/                 # produced by bigbio loader
#      ├── conll2003/              # produced by datasets.load_dataset("eriktks/conll2003")
#      ├── few-nerd/
#      └── cross_ner/{ai,literature,music,politics,science}/

# 2. Regenerate the seeded splits.
python scripts/prepare_seeded_splits.py \
    --raw-root raw_dataset \
    --output-root sampled_dataset \
    --seeds 2026 2027 2028 2029 2030
```

The output directory layout will match `data/seeded_splits/` in this repository.

## License

- **Code and data splits** in this repository are released under [CC-BY-4.0](LICENSE).
- **Original datasets** remain under their respective licenses (see source URLs). Users must comply with the terms of each upstream dataset when using these splits.

## Citation

If you use these splits or scripts, please cite both this repository and the original DHOP paper. A citation entry with a Zenodo DOI is provided in [`CITATION.cff`](CITATION.cff).
