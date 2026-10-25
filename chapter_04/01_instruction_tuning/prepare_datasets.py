#!/usr/bin/env python3
"""Prepare deterministic procedural headline datasets for instruction SFT.

This script is standalone and uses only the Python standard library.  It writes
already split JSON files so a lesson reader only needs to load the selected
profile; no split operation is needed at runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
SPLITS = ("train", "validation", "test")
TASKS = ("exact_copy", "extract_country")

# Each label is an explicit country name and no label is contained in another.
COUNTRIES = (
    "Argentina",
    "Australia",
    "Brazil",
    "Canada",
    "Chile",
    "Colombia",
    "Egypt",
    "India",
    "Japan",
    "Kenya",
    "Mexico",
    "Morocco",
    "New Zealand",
    "Norway",
    "Peru",
    "Portugal",
    "South Africa",
    "Sweden",
    "Thailand",
    "Vietnam",
)

# There are exactly ten short templates: three beginning, four interior, and
# three end placements for the explicit country name.
TEMPLATES = (
    ("beginning", "{country}: students plan a science fair"),
    ("beginning", "{country}: library starts a reading club"),
    ("beginning", "{country}: volunteers plant new trees"),
    ("interior", "Students in {country} plan a science fair"),
    ("interior", "Library staff in {country} start a reading club"),
    ("interior", "Volunteers in {country} plant new trees"),
    ("interior", "A clean-up day in {country} draws neighbors"),
    ("end", "Students plan a science fair in {country}"),
    ("end", "Library staff start a reading club in {country}"),
    ("end", "Volunteers plant new trees in {country}"),
)

INSTRUCTIONS = {
    "exact_copy": (
        "Copy the headline exactly, including punctuation.",
        "Repeat the input headline exactly as written.",
    ),
    "extract_country": (
        "Extract the one explicitly named country from the headline.",
        "Name the single country explicitly written in the headline.",
    ),
}

# This order gives every country a beginning, interior, and end headline early
# in training.  The first seven templates are full-training groups.
TRAIN_TEMPLATE_ORDER = (0, 3, 7, 1, 4, 8, 2)
EVALUATION_TEMPLATE_ORDER = (5, 6, 9)
MAX_HEADLINE_WORDS = 10


def canonical_json(value: Any) -> bytes:
    """Return a stable compact representation used only for checksums."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: Any) -> bytes:
    """Return the human-readable on-disk JSON representation."""
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def country_matches(headline: str) -> list[str]:
    """Find enumerated countries that occur as whole explicit names."""
    return [
        country
        for country in COUNTRIES
        if re.search(r"(?<![A-Za-z])" + re.escape(country) + r"(?![A-Za-z])", headline)
    ]


def stable_group_id(country_index: int, template_index: int) -> str:
    """Produce a readable, fixed identifier from the source grid coordinates."""
    country_slug = COUNTRIES[country_index].lower().replace(" ", "-")
    return f"group-{country_slug}-{template_index + 1:02d}"


def make_groups() -> list[dict[str, Any]]:
    """Build the 20-country by 10-template source grid and its full split."""
    groups: list[dict[str, Any]] = []
    for country_index, country in enumerate(COUNTRIES):
        validation_templates = (
            (5, 6) if country_index % 2 == 0 else (9,)
        )
        for template_index, (position, template) in enumerate(TEMPLATES):
            if template_index in TRAIN_TEMPLATE_ORDER:
                split = "train"
            elif template_index in validation_templates:
                split = "validation"
            else:
                split = "test"
            groups.append(
                {
                    "group_id": stable_group_id(country_index, template_index),
                    "input": template.format(country=country),
                    "country": country,
                    "split": split,
                    # Internal fields are removed before groups.json is written.
                    "_country_index": country_index,
                    "_template_index": template_index,
                    "_position": position,
                }
            )
    return groups


def group_sort_key(group: dict[str, Any]) -> tuple[int, int]:
    return (group["_country_index"], group["_template_index"])


def public_group(group: dict[str, Any]) -> dict[str, str]:
    """Return the deliberately small group schema used by lesson readers."""
    return {
        "group_id": group["group_id"],
        "input": group["input"],
        "country": group["country"],
        "split": group["split"],
    }


def instruction_for(task: str, group: dict[str, Any]) -> str:
    """Select one of the two fixed, seen phrasings without randomness."""
    variant = (group["_country_index"] + group["_template_index"]) % 2
    return INSTRUCTIONS[task][variant]


def records_for_groups(groups: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Create the exact-copy and country-extraction pair for every group."""
    records: list[dict[str, str]] = []
    for group in sorted(groups, key=group_sort_key):
        for task in TASKS:
            records.append(
                {
                    "id": f"{group['group_id']}-{task}",
                    "group_id": group["group_id"],
                    "task": task,
                    "instruction": instruction_for(task, group),
                    "input": group["input"],
                    "output": group["input"] if task == "exact_copy" else group["country"],
                    "country": group["country"],
                }
            )
    return records


def full_profile_groups(all_groups: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: sorted(
            [group for group in all_groups if group["split"] == split], key=group_sort_key
        )
        for split in SPLITS
    }


def short_profile_groups(all_groups: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Round-robin select 64 complete training pairs, retaining shared evaluation."""
    full = full_profile_groups(all_groups)
    train_by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in full["train"]:
        train_by_country[group["country"]].append(group)
    for country in COUNTRIES:
        train_by_country[country].sort(
            key=lambda group: TRAIN_TEMPLATE_ORDER.index(group["_template_index"])
        )

    selected: list[dict[str, Any]] = []
    # Three full rounds produce 60 groups; the next four countries produce 64.
    for round_index in range(4):
        for country in COUNTRIES:
            if len(selected) == 64:
                break
            selected.append(train_by_country[country][round_index])
        if len(selected) == 64:
            break

    return {
        "train": sorted(selected, key=group_sort_key),
        "validation": full["validation"],
        "test": full["test"],
    }


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_source(all_groups: list[dict[str, Any]]) -> None:
    """Validate the source grid, explicit labels, stable IDs, and split accounting."""
    assert_true(len(COUNTRIES) == 20 and len(set(COUNTRIES)) == 20, "need 20 countries")
    assert_true(len(TEMPLATES) == 10, "need 10 templates")
    assert_true(
        Counter(position for position, _ in TEMPLATES)
        == {"beginning": 3, "interior": 4, "end": 3},
        "need beginning, interior, and end template placements",
    )
    for country in COUNTRIES:
        assert_true(
            not any(
                country != other and country in other for other in COUNTRIES
            ),
            f"country label overlaps another label: {country}",
        )

    assert_true(len(all_groups) == 200, "need 200 source groups")
    assert_true(len({group["group_id"] for group in all_groups}) == 200, "duplicate group ID")
    assert_true(len({group["input"] for group in all_groups}) == 200, "duplicate headline")
    for group in all_groups:
        assert_true(
            group["group_id"]
            == stable_group_id(group["_country_index"], group["_template_index"]),
            "unstable group ID",
        )
        assert_true(country_matches(group["input"]) == [group["country"]], "country is not unique")
        assert_true(
            len(re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", group["input"]))
            <= MAX_HEADLINE_WORDS,
            "headline exceeds short token budget",
        )

    split_groups = full_profile_groups(all_groups)
    assert_true(
        {split: len(groups) for split, groups in split_groups.items()}
        == {"train": 140, "validation": 30, "test": 30},
        "full split group counts are wrong",
    )
    split_ids = {split: {group["group_id"] for group in groups} for split, groups in split_groups.items()}
    assert_true(
        not (split_ids["train"] & split_ids["validation"])
        and not (split_ids["train"] & split_ids["test"])
        and not (split_ids["validation"] & split_ids["test"]),
        "source groups cross full splits",
    )
    assert_true(set().union(*split_ids.values()) == {group["group_id"] for group in all_groups}, "full groups missing")

    for country in COUNTRIES:
        counts = {
            split: sum(group["country"] == country for group in split_groups[split])
            for split in SPLITS
        }
        assert_true(counts["train"] == 7, f"{country} must have seven train groups")
        assert_true(counts["validation"] in (1, 2), f"{country} lacks validation coverage")
        assert_true(counts["test"] in (1, 2), f"{country} lacks test coverage")


def validate_profile(
    name: str,
    groups_by_split: dict[str, list[dict[str, Any]]],
    expected_group_counts: dict[str, int],
) -> None:
    """Validate each emitted profile including pair integrity and no duplicate rows."""
    assert_true(set(groups_by_split) == set(SPLITS), f"{name}: missing split")
    assert_true(
        {split: len(groups_by_split[split]) for split in SPLITS} == expected_group_counts,
        f"{name}: incorrect group accounting",
    )
    group_ids_by_split = {
        split: {group["group_id"] for group in groups_by_split[split]} for split in SPLITS
    }
    assert_true(
        sum(len(ids) for ids in group_ids_by_split.values())
        == len(set().union(*group_ids_by_split.values())),
        f"{name}: source group crosses splits",
    )

    all_profile_groups = [group for split in SPLITS for group in groups_by_split[split]]
    for country in COUNTRIES:
        for split in SPLITS:
            assert_true(
                any(group["country"] == country for group in groups_by_split[split]),
                f"{name}: {country} absent from {split}",
            )

    all_records: list[dict[str, str]] = []
    for split in SPLITS:
        records = records_for_groups(groups_by_split[split])
        assert_true(len(records) == 2 * len(groups_by_split[split]), f"{name}: pair count wrong")
        assert_true(len({canonical_json(row) for row in records}) == len(records), f"{name}: duplicate row")
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in records:
            assert_true(
                list(row) == ["id", "group_id", "task", "instruction", "input", "output", "country"],
                f"{name}: row schema wrong",
            )
            assert_true(row["task"] in TASKS, f"{name}: unknown task")
            assert_true(row["instruction"] in INSTRUCTIONS[row["task"]], f"{name}: unknown instruction")
            assert_true(row["id"] == f"{row['group_id']}-{row['task']}", f"{name}: unstable row ID")
            assert_true(country_matches(row["input"]) == [row["country"]], f"{name}: country mismatch")
            expected_output = row["input"] if row["task"] == "exact_copy" else row["country"]
            assert_true(row["output"] == expected_output, f"{name}: wrong target")
            by_group[row["group_id"]].append(row)
        assert_true(set(by_group) == group_ids_by_split[split], f"{name}: records and groups differ")
        for group_id, pair in by_group.items():
            assert_true(len(pair) == 2, f"{name}: incomplete pair {group_id}")
            assert_true({row["task"] for row in pair} == set(TASKS), f"{name}: task pair wrong")
            assert_true(len({row["input"] for row in pair}) == 1, f"{name}: pair input mismatch")
            assert_true(len({row["country"] for row in pair}) == 1, f"{name}: pair country mismatch")
        all_records.extend(records)
    assert_true(len({row["id"] for row in all_records}) == len(all_records), f"{name}: duplicate ID")

    # Both instruction forms are deliberately present in every training profile.
    training_records = records_for_groups(groups_by_split["train"])
    for task in TASKS:
        assert_true(
            {row["instruction"] for row in training_records if row["task"] == task}
            == set(INSTRUCTIONS[task]),
            f"{name}: not all seen instruction variants occur in train",
        )


def validate_profiles(
    full: dict[str, list[dict[str, Any]]], short: dict[str, list[dict[str, Any]]]
) -> None:
    validate_profile("full", full, {"train": 140, "validation": 30, "test": 30})
    validate_profile("short", short, {"train": 64, "validation": 30, "test": 30})

    full_train_ids = {group["group_id"] for group in full["train"]}
    short_train_ids = {group["group_id"] for group in short["train"]}
    assert_true(short_train_ids <= full_train_ids, "short training is not a full-training subset")
    assert_true(len(short_train_ids) == 64, "short training must contain 64 groups")
    country_counts = Counter(group["country"] for group in short["train"])
    assert_true(set(country_counts) == set(COUNTRIES), "short training misses a country")
    assert_true(max(country_counts.values()) - min(country_counts.values()) <= 1, "short training is not balanced")
    for split in ("validation", "test"):
        assert_true(
            [public_group(group) for group in short[split]]
            == [public_group(group) for group in full[split]],
            f"{split} is not shared across profiles",
        )
        assert_true(
            records_for_groups(short[split]) == records_for_groups(full[split]),
            f"{split} records are not shared across profiles",
        )


def country_split_counts(groups_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    return {
        country: {
            split: sum(group["country"] == country for group in groups_by_split[split])
            for split in SPLITS
        }
        for country in COUNTRIES
    }


def write_json(path: Path, value: Any) -> str:
    """Write one owned generated JSON file and return its on-disk SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json_bytes(value)
    path.write_bytes(content)
    return sha256_bytes(content)


def build_profile_files(name: str, groups_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Write a profile's three record files and its active group catalog."""
    profile_root = DATA_ROOT / name
    file_info: dict[str, dict[str, Any]] = {}
    all_public_groups: list[dict[str, str]] = []
    for split in SPLITS:
        records = records_for_groups(groups_by_split[split])
        record_path = profile_root / f"{split}.json"
        file_info[split] = {
            "relative_path": f"{name}/{split}.json",
            "sha256": write_json(record_path, records),
            "rows": len(records),
            "groups": len(groups_by_split[split]),
        }
        all_public_groups.extend(public_group(group) for group in groups_by_split[split])
    groups_path = profile_root / "groups.json"
    file_info["groups"] = {
        "relative_path": f"{name}/groups.json",
        "sha256": write_json(groups_path, all_public_groups),
        "rows": len(all_public_groups),
        "groups": len(all_public_groups),
    }
    return {
        "group_counts": {split: len(groups_by_split[split]) for split in SPLITS},
        "row_counts": {split: 2 * len(groups_by_split[split]) for split in SPLITS},
        "country_group_counts": country_split_counts(groups_by_split),
        "files": file_info,
    }


def make_manifest(profile_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "purpose": "Beginner English headline instruction SFT with exact copying and country extraction.",
        "source_description": "Clean procedural practice headlines generated from a fixed grid; they are not real news.",
        "preparation": "JSON contains clean labels and fixed splits, not model token IDs.",
        "tokenization": "Tokenize after loading with the chosen model vocabulary; no model-specific pretokenization is stored.",
        "determinism": {
            "randomness": "none",
            "group_id_rule": "group-{lowercase-country-with-hyphens}-{one-based-template-index}",
            "row_id_rule": "{group_id}-{task}",
            "full_split_rule": "Seven groups per country train; the remaining three alternate 2 validation/1 test and 1 validation/2 test by country order.",
            "short_train_rule": "Round-robin first 64 full-training groups by country, retaining complete two-task pairs.",
        },
        "schema": {
            "record_fields": ["id", "group_id", "task", "instruction", "input", "output", "country"],
            "group_fields": ["group_id", "input", "country", "split"],
            "tasks": list(TASKS),
            "prompt_fields_only": ["instruction", "input"],
            "prompt_excludes": ["id", "group_id", "task", "country"],
        },
        "inventory": {
            "countries": list(COUNTRIES),
            "headline_templates": [
                {"country_position": position, "template": template}
                for position, template in TEMPLATES
            ],
            "source_groups": 200,
            "source_rows": 400,
            "max_headline_words": MAX_HEADLINE_WORDS,
        },
        "profiles": profile_info,
        "shared_evaluation": "The short and full profiles use identical validation and test groups and rows.",
    }


def main() -> None:
    all_groups = make_groups()
    full = full_profile_groups(all_groups)
    short = short_profile_groups(all_groups)
    validate_source(all_groups)
    validate_profiles(full, short)

    profiles = {
        "short": build_profile_files("short", short),
        "full": build_profile_files("full", full),
    }
    manifest_path = DATA_ROOT / "manifest.json"
    write_json(manifest_path, make_manifest(profiles))

    print("full: train=140 groups/280 rows, validation=30/60, test=30/60")
    print("short: train=64 groups/128 rows, validation=30/60, test=30/60")
    print("wrote data/{short,full}/{train,validation,test,groups}.json and data/manifest.json")


if __name__ == "__main__":
    main()
