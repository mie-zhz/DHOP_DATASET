"""Seeded raw-dataset sampling for supplementary NER experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from datasets import DatasetDict, load_from_disk


DATASETS = [
    "bc5cdr",
    "conll2003",
    "cross_ner_ai",
    "cross_ner_literature",
    "cross_ner_music",
    "cross_ner_politics",
    "cross_ner_science",
    "few-nerd",
]

DATASET_DIRS = {
    "bc5cdr": "bc5cdr",
    "conll2003": "conll2003",
    "few-nerd": "few-nerd",
    "cross_ner_ai": "cross_ner/ai",
    "cross_ner_literature": "cross_ner/literature",
    "cross_ner_music": "cross_ner/music",
    "cross_ner_politics": "cross_ner/politics",
    "cross_ner_science": "cross_ner/science",
}

FIXED_ENTITY_TYPES = {
    "bc5cdr": ["Chemical", "Disease"],
    "conll2003": ["LOC", "ORG", "PER"],
    "few-nerd": ["art", "location", "organization", "person", "product"],
    "cross_ner_ai": ["algorithm", "field", "product", "researcher", "task"],
    "cross_ner_literature": ["book", "literarygenre", "location", "person", "writer"],
    "cross_ner_music": ["album", "award", "band", "musicalartist", "musicgenre"],
    "cross_ner_politics": ["election", "location", "organisation", "politicalparty", "politician"],
    "cross_ner_science": ["astronomicalobject", "chemicalcompound", "organisation", "person", "scientist"],
}


def dataset_rel_dir(dataset: str) -> Path:
    return Path(DATASET_DIRS[dataset])


def raw_dataset_path(raw_root: Path, dataset: str) -> Path:
    return raw_root / dataset_rel_dir(dataset)


def output_dataset_path(output_root: Path, dataset: str) -> Path:
    return output_root / dataset_rel_dir(dataset)


def convert_dataset(dataset: str, raw_root: Path) -> list[dict]:
    ds = load_from_disk(str(raw_dataset_path(raw_root, dataset)))
    entity_types = set(FIXED_ENTITY_TYPES[dataset])
    if dataset == "bc5cdr":
        records = _convert_bc5cdr(ds, entity_types)
    else:
        records = _convert_token_dataset(ds, entity_types, dataset)
    return records


def _iter_split_rows(ds: DatasetDict) -> Iterable[tuple[str, int, dict]]:
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for idx, row in enumerate(ds[split]):
            yield split, idx, row


def _convert_bc5cdr(ds: DatasetDict, entity_types: set[str]) -> list[dict]:
    converted = []
    for split, idx, row in _iter_split_rows(ds):
        passages = row.get("passages") or []
        text_parts = []
        entities = []
        offset_base = 0
        for passage in passages:
            if text_parts:
                text_parts.append(" ")
                offset_base += 1
            passage_text = passage.get("text") or ""
            text_parts.append(passage_text)
            for entity in passage.get("entities") or []:
                label = entity.get("type")
                if label not in entity_types:
                    continue
                text_values = entity.get("text") or []
                offsets = entity.get("offsets") or []
                if not text_values or not offsets:
                    continue
                start, end = offsets[0]
                entities.append(
                    {
                        "text": str(text_values[0]),
                        "label": label,
                        "start": int(start) + offset_base,
                        "end": int(end) + offset_base,
                    }
                )
            offset_base += len(passage_text)
        converted.append(
            {
                "text": "".join(text_parts),
                "entities": entities,
                "_raw_split": split,
                "_raw_index": idx,
            }
        )
    return converted


def _convert_token_dataset(ds: DatasetDict, entity_types: set[str], dataset: str) -> list[dict]:
    converted = []
    for split, idx, row in _iter_split_rows(ds):
        tokens = [str(t) for t in row.get("tokens", [])]
        tags = row.get("ner_tags", [])
        tag_feature = ds[split].features["ner_tags"].feature
        labels = [tag_feature.int2str(int(tag)) for tag in tags]
        text, spans = _join_tokens_with_spans(tokens)
        entities = _bio_entities(tokens, spans, labels, entity_types, dataset)
        converted.append(
            {
                "text": text,
                "entities": entities,
                "_raw_split": split,
                "_raw_index": idx,
            }
        )
    return converted


def _join_tokens_with_spans(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text_parts = []
    spans = []
    cursor = 0
    no_space_before = {".", ",", ":", ";", "!", "?", ")", "]", "}", "%", "''", "'s", "n't"}
    no_space_after = {"(", "[", "{", "``", "$"}
    prev = ""
    for token in tokens:
        add_space = bool(text_parts) and token not in no_space_before and prev not in no_space_after
        if add_space:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        spans.append((start, cursor))
        prev = token
    return "".join(text_parts), spans


def _bio_entities(
    tokens: list[str],
    spans: list[tuple[int, int]],
    labels: list[str],
    entity_types: set[str],
    dataset: str,
) -> list[dict]:
    entities = []
    current_type = None
    current_start = None
    current_end = None

    def flush():
        nonlocal current_type, current_start, current_end
        if current_type and current_type in entity_types and current_start is not None and current_end is not None:
            entities.append(
                {
                    "text": text[current_start:current_end],
                    "label": current_type,
                    "start": current_start,
                    "end": current_end,
                }
            )
        current_type = None
        current_start = None
        current_end = None

    text, _ = _join_tokens_with_spans(tokens)
    for label, (start, end) in zip(labels, spans):
        if label == "O":
            flush()
            continue
        prefix, ent_type = _split_label(label, dataset)
        if ent_type not in entity_types:
            flush()
            continue
        if prefix == "B" or current_type != ent_type:
            flush()
            current_type = ent_type
            current_start = start
            current_end = end
        else:
            current_end = end
    flush()
    return entities


def _split_label(label: str, dataset: str) -> tuple[str, str]:
    if dataset == "few-nerd":
        return "I", label
    if "-" not in label:
        return "B", label
    prefix, ent_type = label.split("-", 1)
    return prefix, ent_type


def write_seeded_splits(
    *,
    dataset: str,
    seed: int,
    raw_root: Path,
    output_root: Path,
    runtime_samples: int = 200,
    blind_samples: int = 500,
) -> dict:
    records = convert_dataset(dataset, raw_root)
    required = runtime_samples + blind_samples
    if len(records) < required:
        raise ValueError(f"{dataset} has {len(records)} records, need {required}")

    rng = random.Random(seed)
    selected = rng.sample(range(len(records)), required)
    runtime_indices = selected[:runtime_samples]
    blind_indices = selected[runtime_samples:]

    runtime_all = [_strip_raw(records[i]) for i in runtime_indices]
    blind_test = [_strip_raw(records[i]) for i in blind_indices]

    out_dir = output_dataset_path(output_root, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "runtime_all.json", runtime_all)
    _write_json(out_dir / "blind_test.json", blind_test)
    _write_json(out_dir / "runtime_train_test_train.json", runtime_all[:100])
    _write_json(out_dir / "runtime_train_test_test.json", runtime_all[100:200])
    _write_json(out_dir / "runtime_train.json", runtime_all[:80])
    _write_json(out_dir / "runtime_dev.json", runtime_all[80:140])
    _write_json(out_dir / "runtime_test.json", runtime_all[140:200])

    manifest = {
        "dataset": dataset,
        "seed": seed,
        "raw_root": str(raw_root),
        "output_dir": str(out_dir),
        "entity_types": FIXED_ENTITY_TYPES[dataset],
        "runtime_samples": runtime_samples,
        "blind_samples": blind_samples,
        "runtime_raw_indices": [_raw_ref(records[i]) for i in runtime_indices],
        "blind_raw_indices": [_raw_ref(records[i]) for i in blind_indices],
    }
    _write_json(out_dir / "sample_manifest.json", manifest)
    return manifest


def _strip_raw(record: dict) -> dict:
    return {"text": record["text"], "entities": record.get("entities", [])}


def _raw_ref(record: dict) -> dict:
    return {"split": record.get("_raw_split"), "index": record.get("_raw_index")}


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
