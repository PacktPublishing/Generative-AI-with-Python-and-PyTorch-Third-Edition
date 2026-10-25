"""Supporting functions for the instruction-SFT training notebook.

The notebook keeps the model and Trainer setup visible. Data integrity, padding,
generation, scoring, and optional resource measurements live here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

# Plain-text progress does not need optional Jupyter widgets.
warnings.filterwarnings("ignore", message="IProgress not found.*")
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import psutil
import torch
from datasets import Dataset, DatasetDict
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MAX_LENGTH = 128
MAX_NEW_TOKENS = 64
PROFILES = {
    "short": {"epochs": 1, "train_groups": 64, "train_rows": 128, "batch_size": 4},
    "full": {"epochs": 3, "train_groups": 140, "train_rows": 280, "batch_size": 4},
}
TASKS = ("exact_copy", "extract_country")


def select_device():
    """Select CPU/MPS before constructing Trainer; allow a test-run override."""
    requested = os.environ.get("SFT_DEVICE", "auto")
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("SFT_DEVICE must be auto, cpu, or mps")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if requested == "auto" else requested
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device


def load_profile(profile):
    """Load already prepared files. No labels/splits are constructed at runtime."""
    if profile not in PROFILES:
        raise ValueError("Select profile 'short' or 'full'")
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    specification = manifest["profiles"][profile]
    raw = {}
    for name, info in specification["files"].items():
        path = ROOT / "data" / info["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {path.name}")
        raw[name] = json.loads(path.read_text())
    # The builder validates labels and group-aware splits once. Checksums above
    # ensure these are still exactly the prepared files, avoiding duplicate checks.
    groups = raw.pop("groups")
    return DatasetDict({name: Dataset.from_list(rows) for name, rows in raw.items()}), groups


def data_summary(data, groups):
    counts = Counter(g["split"] for g in groups)
    return pd.DataFrame([
        {"Split": split, "Headlines": counts[split], "Examples": len(data[split]),
         "Copy": sum(r["task"] == "exact_copy" for r in data[split]),
         "Extract": sum(r["task"] == "extract_country" for r in data[split])}
        for split in ("train", "validation", "test")
    ]).set_index("Split")


def show_examples(rows, number_of_groups=2):
    selected = []
    for row in rows:
        if row["group_id"] not in selected:
            selected.append(row["group_id"])
            if len(selected) == number_of_groups:
                break
    return pd.DataFrame([
        {"Instruction": r["instruction"], "Input": r["input"], "Expected response": r["output"]}
        for r in rows if r["group_id"] in selected
    ])


def make_prompt(instruction, text):
    return f"###Instruction:\n{instruction}\n###Input:\n{text}\n###Response:"


def encode_example(row, tokenizer):
    prompt = tokenizer.encode(make_prompt(row["instruction"], row["input"]), add_special_tokens=False)
    response = tokenizer.encode(row["output"], add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt + response
    if len(ids) > MAX_LENGTH:
        raise ValueError("Complete example exceeds the token budget; targets are never truncated")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt) + response}


def prepare_training_data(rows, tokenizer):
    return Dataset.from_list([encode_example(row, tokenizer) for row in rows])


def show_response_boundary(row, tokenizer):
    encoded = encode_example(row, tokenizer)
    boundary = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
    return pd.DataFrame([
        {"Part": "Prompt (read as context)", "Text": tokenizer.decode(encoded["input_ids"][:boundary]),
         "Labels": "-100 (ignored)", "Token count": boundary},
        {"Part": "Response + EOS (predicted)", "Text": tokenizer.decode(encoded["input_ids"][boundary:]),
         "Labels": "Actual response/EOS IDs", "Token count": len(encoded["input_ids"]) - boundary},
    ])


class ResponseOnlyCollator:
    """Right-pad inputs; preserve the response labels supplied by preprocessing."""
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, examples):
        width = max(len(r["input_ids"]) for r in examples)
        return {
            key: torch.tensor([r[key] + [pad] * (width - len(r[key])) for r in examples],
                              dtype=torch.long)
            for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100))
        }


class RunMonitor(TrainerCallback):
    """Sample current memory/time; these samples do not claim allocator peaks."""
    def __init__(self, device):
        self.device = device
        self.samples = []
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_epoch_end(self, args, state, control, **kwargs):
        sample = {"epoch": state.epoch, "step": state.global_step,
                  "elapsed_seconds": time.monotonic() - self.started,
                  "cpu_rss_bytes": psutil.Process().memory_info().rss}
        if self.device == "mps":
            sample["mps_current_bytes"] = torch.mps.current_allocated_memory()
            sample["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        self.samples.append(sample)


def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def save_run_summary(directory, profile, device, trainer, train_result, validation, monitor):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = trainer.state.best_model_checkpoint
    record = {
        "profile": profile, "device": device, "model": MODEL_NAME, "revision": MODEL_REVISION,
        "train_examples": PROFILES[profile]["train_rows"], "validation_examples": 60,
        "test_examples": 60, "updates": trainer.state.global_step,
        "epochs": train_result.metrics["epoch"], "training_metrics": train_result.metrics,
        "validation_metrics": validation,
        "selected_checkpoint": Path(checkpoint).name if checkpoint else None,
        "memory_samples": monitor.samples,
    }
    expected = PROFILES[profile]["epochs"] * PROFILES[profile]["train_rows"] // PROFILES[profile]["batch_size"]
    if record["updates"] != expected or not all(torch.isfinite(torch.tensor(v)) for v in [
        train_result.training_loss, validation["eval_loss"]
    ]):
        raise RuntimeError("Unexpected training update count or non-finite loss")
    (directory / "run_summary.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def generate_responses(model, tokenizer, rows, device, batch_size=8):
    """Identical greedy settings, left-padded prompts, continuation-only decoding."""
    model.eval()
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions = []
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = [rows[i] for i in range(start, min(start + batch_size, len(rows)))]
            encoded = tokenizer([make_prompt(r["instruction"], r["input"]) for r in batch_rows],
                                padding=True, return_tensors="pt").to(device)
            width = encoded["input_ids"].shape[1]
            if width + MAX_NEW_TOKENS > model.config.n_positions:
                raise ValueError("Prompt plus generation exceeds model context")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            for row, ids in zip(batch_rows, generated[:, width:].cpu().tolist()):
                eos = ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in ids else None
                continuation = ids if eos is None else ids[:eos + 1]
                text = tokenizer.decode(continuation, skip_special_tokens=True)
                predictions.append({
                    **{k: row[k] for k in ("id", "group_id", "task", "instruction", "input", "output")},
                    "prediction": text,
                    "cap_hit": eos is None and len(continuation) == MAX_NEW_TOKENS,
                })
            del encoded, generated
    finally:
        tokenizer.padding_side = previous_padding
    return predictions


def compare_saved_model(directory, test_rows, device):
    """Load the tuned and untouched starting checkpoints one at a time."""
    directory = Path(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                               clean_up_tokenization_spaces=False)
    tuned_model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
                                                       dtype=torch.float32, attn_implementation="eager")
    tuned_model.to(device)
    tuned = generate_responses(tuned_model, tokenizer, test_rows, device)
    del tuned_model
    clear_memory()
    # Reset before PAD embedding initialization for the baseline comparison.
    from transformers import set_seed
    set_seed(42)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                                     dtype=torch.float32, attn_implementation="eager")
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.to(device)
    baseline = generate_responses(base_model, tokenizer, test_rows, device)
    del base_model
    clear_memory()
    if [r["id"] for r in baseline] != [r["id"] for r in tuned]:
        raise RuntimeError("Comparison example ordering differs")
    return baseline, tuned


def comparison_table(baseline, tuned, number_of_groups=3):
    by_id = {r["id"]: r for r in baseline}
    selected = []
    for row in tuned:
        if row["group_id"] not in selected:
            selected.append(row["group_id"])
            if len(selected) == number_of_groups:
                break
    return pd.DataFrame([
        {"Group": r["group_id"], "Input": r["input"], "Instruction": r["instruction"], "Expected": r["output"],
         "Starting model": by_id[r["id"]]["prediction"], "Tuned model": r["prediction"]}
        for r in tuned if r["group_id"] in selected
    ])


def score_predictions(predictions):
    grouped = defaultdict(dict)
    result = {}
    for row in predictions:
        if row["task"] in grouped[row["group_id"]]:
            raise ValueError("Repeated task in a source group")
        grouped[row["group_id"]][row["task"]] = row
    if not grouped or any(set(pair) != set(TASKS) for pair in grouped.values()):
        raise ValueError("Evaluation needs complete source/task pairs")
    for task in TASKS:
        rows = [r for r in predictions if r["task"] == task]
        correct = sum(r["prediction"].strip() == r["output"].strip() for r in rows)
        result[task] = {"correct": correct, "total": len(rows), "rate": correct / len(rows)}
    correct_pairs = sum(all(r["prediction"].strip() == r["output"].strip() for r in pair.values())
                        for pair in grouped.values())
    switched = sum(pair["exact_copy"]["prediction"].strip() != pair["extract_country"]["prediction"].strip()
                   for pair in grouped.values())
    result["both_correct"] = {"correct": correct_pairs, "total": len(grouped), "rate": correct_pairs / len(grouped)}
    result["different_responses"] = {"correct": switched, "total": len(grouped), "rate": switched / len(grouped)}
    copies = [r for r in predictions if r["task"] == "exact_copy"]
    hypotheses = [r["prediction"] for r in copies]
    references = [[r["output"] for r in copies]]
    bleu, chrf = BLEU(tokenize="13a", lowercase=False), CHRF(char_order=6, word_order=0, beta=2)
    result["copy_bleu"] = bleu.corpus_score(hypotheses, references).score
    result["copy_chrf"] = chrf.corpus_score(hypotheses, references).score
    result["bleu_signature"], result["chrf_signature"] = str(bleu.get_signature()), str(chrf.get_signature())
    result["blank_rate"] = sum(not r["prediction"].strip() for r in predictions) / len(predictions)
    result["cap_hit_rate"] = sum(r["cap_hit"] for r in predictions) / len(predictions)
    return result


def scorecard(baseline, tuned):
    scores = {"starting": score_predictions(baseline), "tuned": score_predictions(tuned)}
    labels = {
        "exact_copy": "Copy exact match", "extract_country": "Country exact match",
        "both_correct": "Both correct per headline", "different_responses": "Different responses (diagnostic)",
    }
    table = []
    for key, label in labels.items():
        table.append({"Measure": label, **{
            name: f"{values[key]['correct']}/{values[key]['total']} ({values[key]['rate']:.0%})"
            for name, values in scores.items()
        }})
    for key, label in (("copy_bleu", "Copy BLEU (overlap)"), ("copy_chrf", "Copy chrF (overlap)")):
        table.append({"Measure": label, **{name: f"{values[key]:.2f}" for name, values in scores.items()}})
    return pd.DataFrame(table).set_index("Measure"), scores


def save_evaluation(directory, baseline, tuned, scores):
    path = Path(directory) / "evaluation.json"
    path.write_text(json.dumps({"scores": scores, "baseline_predictions": baseline,
                                "tuned_predictions": tuned}, ensure_ascii=False, indent=2) + "\n")
    return path
