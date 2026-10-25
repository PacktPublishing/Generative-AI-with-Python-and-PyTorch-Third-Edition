"""Small explicit utilities for the pinned Aesop DPO exercise."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_LENGTH = 768
BETA = 0.1
PROFILE_STEPS = {'short': 2, 'full': 96}
STYLE_TARGET = 'Standalone Markdown-bold headline, a blank line, and a short source-faithful summary.'
_FIELDS = {"id", "title", "passage", "chosen", "rejected", "preference_reason",
           "rejection_type", "essential_facts", "label_origin", "author_model", "group_id"}


def choose_device(device_choice="auto"):
    if device_choice not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device_choice must be auto, cpu, mps, or cuda")
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda = torch.cuda.is_available()
    if device_choice == "auto":
        return "cuda" if cuda else ("mps" if mps else "cpu")
    if device_choice == "cuda" and not cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_choice == "mps" and not mps:
        raise RuntimeError("MPS was requested but is unavailable")
    return device_choice


def load_base(seed=42, num_threads=4):
    """Load the exact float32 eager base; callers own device placement."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager")
    model.config.use_cache = False
    return model, tokenizer


def render_prompt(tokenizer, passage):
    """Ask only for a concise faithful summary; style is learned from preferences."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You summarize the supplied passage faithfully."},
         {"role": "user", "content": "Provide a concise, faithful summary of the main events and outcome. Do not invent facts.\n\nPassage:\n" + passage}],
        tokenize=False, add_generation_prompt=True)


def style_flags(text):
    """Describe output structure only; this deliberately makes no factual judgment."""
    text = text if isinstance(text, str) else ""
    lines = text.splitlines()
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    whole_body = text.strip()
    result = {'bold_headline': False, 'headline_summary_format': False,
              'headline': None, 'body': whole_body, 'word_count': len(text.split())}
    if first_index is None:
        return result
    first = lines[first_index]
    match = re.fullmatch(r'\*\*([^*\n]+)\*\*', first)
    if not match:
        return result
    headline = match.group(1)
    if (not headline.strip() or not 1 <= len(headline.split()) <= 12
            or headline.lstrip().startswith(('*', '#', '`'))):
        return result
    # A bold title must occupy the first nonblank line; embedded prose is not one.
    result.update({'bold_headline': True, 'headline': headline, 'body': '\n'.join(lines[first_index + 1:]).strip()})
    body_lines = lines[first_index + 1:]
    blank_after_headline = bool(body_lines) and not body_lines[0].strip()
    body = '\n'.join(body_lines[1:]).strip() if blank_after_headline else ''
    has_extra_headline = any(re.fullmatch(r'\*\*([^*\n]+)\*\*', line) for line in body_lines)
    has_code_fence = any('```' in line for line in body_lines)
    result['headline_summary_format'] = bool(blank_after_headline and body and not has_extra_headline and not has_code_fence)
    return result


def score_styles(generations):
    return [{'id': row['id'], **style_flags(row['summary'])} for row in generations]


def summarize_styles(records):
    if not records:
        raise ValueError('cannot summarize no style records')
    count = len(records)
    return {'sample_count': count,
            'bold_headline_rate': sum(bool(row['bold_headline']) for row in records) / count,
            'headline_summary_format_rate': sum(bool(row['headline_summary_format']) for row in records) / count,
            'mean_output_words': sum(row['word_count'] for row in records) / count}


def _ids(tokenizer, text):
    value = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(value[0] if value and isinstance(value[0], list) else value)


def validate_and_tokenize(tokenizer, prompt, response, max_length=MAX_LENGTH):
    if not isinstance(prompt, str) or not prompt or not isinstance(response, str) or not response.strip():
        raise ValueError("prompt and response must be nonempty strings")
    if "<|im_" in response:
        raise ValueError("response must be plain text, not chat markup")
    if not tokenizer.eos_token:
        raise ValueError("tokenizer has no EOS token")
    prompt_ids = _ids(tokenizer, prompt)
    full_ids = _ids(tokenizer, prompt + response + tokenizer.eos_token)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not an exact token prefix of prompt+completion")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("completion tokenization is empty")
    if full_ids[-1] != tokenizer.eos_token_id:
        raise ValueError("the appended EOS was not retained as the final token")
    if len(full_ids) > max_length:
        raise ValueError("example exceeds max_length; refusing to truncate")
    return full_ids, len(prompt_ids)


def _manifest_expected(manifest, filename):
    item = manifest.get("files", {}).get(filename)
    if isinstance(item, dict): return item.get("sha256")
    return item if isinstance(item, str) else None


def load_splits(data_dir=Path("data")):
    data_dir = Path(data_dir)
    manifest_file = data_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding='utf-8')) if manifest_file.exists() else None
    result, all_ids, groups, passages = {}, set(), set(), set()
    for split in ("train", "validation", "test"):
        filename = split + ".json"; raw = (data_dir / filename).read_bytes()
        if manifest:
            expected = _manifest_expected(manifest, filename)
            if expected and hashlib.sha256(raw).hexdigest() != expected.lower():
                raise ValueError(f"{filename} does not match manifest checksum")
        rows = json.loads(raw)
        if not isinstance(rows, list) or not rows: raise ValueError(f"{filename} must be a nonempty list")
        for row in rows:
            if not isinstance(row, dict) or not _FIELDS.issubset(row):
                raise ValueError("invalid preference record fields")
            if (not isinstance(row['essential_facts'], list)
                    or len(row['essential_facts']) not in (2, 3)
                    or any(not isinstance(fact, str) or not fact.strip() for fact in row['essential_facts'])):
                raise ValueError('essential_facts must contain two or three nonempty source facts')
            if any(not isinstance(row[x], str) or not row[x].strip() for x in _FIELDS - {"essential_facts"}):
                raise ValueError("preference record has empty required text")
            if row["id"] in all_ids or row["group_id"] in groups or row["passage"] in passages:
                raise ValueError("id, group_id, and exact passage must not leak between splits")
            # This corpus has no reviewed human labels. Do not represent AI labels as human.
            if row['label_origin'] != 'AI-authored synthetic preference; not human-reviewed':
                raise ValueError('This dataset requires explicitly AI-authored, non-human-reviewed labels')
            all_ids.add(row["id"]); groups.add(row["group_id"]); passages.add(row["passage"])
        result[split] = rows
    return result


def load_data_bundle(data_dir=Path("data")):
    """Load validated splits and their manifest-backed content provenance."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing {manifest_path}; run prepare_data.py to create the data bundle")
    raw_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json is malformed; run prepare_data.py to recreate the data bundle") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object; run prepare_data.py")
    if not isinstance(manifest.get('files'), dict):
        raise ValueError("manifest.json files must contain split checksums; run prepare_data.py")
    split_hashes = {}
    split_sizes = {}
    for split in ("train", "validation", "test"):
        filename = split + ".json"
        path = data_dir / filename
        if not path.is_file():
            raise ValueError(f"Missing {path}; run prepare_data.py to create the data bundle")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        expected = _manifest_expected(manifest, filename)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError(f"manifest.json is missing a valid SHA-256 checksum for {filename}")
        if actual != expected.lower():
            raise ValueError(f"{filename} does not match manifest checksum; run prepare_data.py")
        try:
            rows = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{filename} is malformed; run prepare_data.py") from exc
        if not isinstance(rows, list):
            raise ValueError(f"{filename} must contain a list")
        split_sizes[split] = len(rows)
        split_hashes[filename] = actual
    if manifest.get("split_sizes") != split_sizes:
        raise ValueError(f"manifest split_sizes must exactly match actual counts: {split_sizes}")
    splits = load_splits(data_dir)
    origins = sorted({row["label_origin"] for rows in splits.values() for row in rows})
    models = sorted({row["author_model"] for rows in splits.values() for row in rows})
    return splits, {"manifest": manifest,
                    "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "split_sha256": split_hashes,
                    "label_origins": origins,
                    "author_models": models}


def make_dataset(rows, tokenizer, max_length=MAX_LENGTH):
    from datasets import Dataset
    out = []
    for row in rows:
        prompt = render_prompt(tokenizer, row["passage"])
        validate_and_tokenize(tokenizer, prompt, row["chosen"], max_length)
        validate_and_tokenize(tokenizer, prompt, row["rejected"], max_length)
        out.append({"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]})
    return Dataset.from_list(out)


def _device(model):
    try: return next(model.parameters()).device
    except StopIteration: return torch.device("cpu")


def score_response(model, tokenizer, prompt, response, max_length=MAX_LENGTH):
    ids, prompt_length = validate_and_tokenize(tokenizer, prompt, response, max_length)
    completion = len(ids) - prompt_length
    input_ids = torch.tensor([ids], dtype=torch.long, device=_device(model))
    old = model.training; model.eval()
    try:
        with torch.inference_mode():
            # Qwen can return just the needed tail logits.  Other causal LMs use
            # the ordinary full-logit path rather than concealing an API error.
            try:
                logits = model(input_ids=input_ids, logits_to_keep=completion + 1, use_cache=False).logits
                states = logits[:, :-1, :]
                if states.shape[1] != completion: raise RuntimeError("unexpected logits_to_keep shape")
            except TypeError:
                logits = model(input_ids=input_ids).logits
                states = logits[:, -(completion + 1):-1, :]
            targets = input_ids[:, -completion:]
            chosen = states.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            logp = (chosen - torch.logsumexp(states, dim=-1)).sum()
            answer = float(logp.item())
    finally:
        model.train(old)
    if not math.isfinite(answer): raise ValueError("non-finite completion log probability")
    return answer


def _disable_adapter(model):
    """PEFT's public adapter-off context; a plain base is its own reference."""
    method = getattr(model, "disable_adapter", None)
    return method() if method is not None else contextlib.nullcontext()


def score_pairs(model, tokenizer, rows, beta=BETA, reference_scores=None, max_length=MAX_LENGTH):
    if beta <= 0: raise ValueError("beta must be positive")
    results = []
    for row in rows:
        prompt = render_prompt(tokenizer, row["passage"])
        chosen, rejected = score_response(model, tokenizer, prompt, row["chosen"], max_length), score_response(model, tokenizer, prompt, row["rejected"], max_length)
        cached = (reference_scores or {}).get(row["id"])
        if cached is None:
            with _disable_adapter(model):
                rc, rr = score_response(model, tokenizer, prompt, row["chosen"], max_length), score_response(model, tokenizer, prompt, row["rejected"], max_length)
        else:
            rc, rr = cached["ref_chosen_logp"], cached["ref_rejected_logp"]
        margin = beta * (chosen - rc - rejected + rr)
        results.append({"id": row["id"], "chosen_logp": chosen, "rejected_logp": rejected,
                        "ref_chosen_logp": rc, "ref_rejected_logp": rr, "margin": margin,
                        "dpo_loss": float(torch.nn.functional.softplus(torch.tensor(-margin)).item())})
    return results


def summarize_pairs(scores):
    if not scores: raise ValueError("cannot summarize no scores")
    margins = [x["margin"] for x in scores]; tol = 1e-6
    ties = sum(abs(x) <= tol for x in margins)
    preferred = sum(1 if x > tol else .5 if abs(x) <= tol else 0 for x in margins)
    return {"mean_margin": sum(margins)/len(margins), "mean_dpo_loss": sum(x["dpo_loss"] for x in scores)/len(scores),
            "preferred_fraction": preferred/len(scores), "tie_fraction": ties/len(scores), "sample_count": len(scores)}


def generate_summaries(model, tokenizer, rows, max_new_tokens=80, reference=False):
    old = model.training; model.eval(); out = []
    ctx = _disable_adapter(model) if reference else contextlib.nullcontext()
    try:
        with ctx, torch.inference_mode():
            for row in rows:
                prompt = render_prompt(tokenizer, row["passage"])
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(_device(model))
                generated = model.generate(**encoded, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                new_ids = generated[0, encoded["input_ids"].shape[1]:]
                out.append({"id": row["id"], "title": row["title"], "passage": row["passage"],
                            "summary": tokenizer.decode(new_ids, skip_special_tokens=True)})
    finally: model.train(old)
    return out


try:  # Keep this small module importable in tokenizer-only unit environments.
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object


class TrainingMonitor(_TrainerCallback):
    """TRL/Transformers callback storing only finite numeric training telemetry."""
    history = None
    def __init__(self): self.history = []; self._start = time.monotonic()
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}; allowed = {"loss", "eval_loss", "rewards/margins", "rewards/accuracies", "rewards/chosen", "rewards/rejected", "grad_norm", "learning_rate"}
        item = {k: float(v) for k, v in logs.items() if k in allowed and isinstance(v, (int,float))}
        if any(not math.isfinite(v) for v in item.values()): raise ValueError("non-finite trainer metric")
        if not item:
            return control
        item["step"] = state.global_step; item["elapsed_seconds"] = time.monotonic()-self._start
        try:
            import psutil; item["rss_bytes"] = psutil.Process().memory_info().rss
        except ImportError: pass
        if item: self.history.append(item)
        return control


class BaseWeightSnapshot:
    """A cheap guard that records base-weight values without sampling LoRA tensors."""
    def __init__(self, model, count=16):
        for name, parameter in model.named_parameters():
            if "lora_" not in name.lower() and parameter.ndim >= 2:
                self.name = name
                self.values = parameter.detach().reshape(-1)[:count].cpu().clone()
                return
        raise ValueError("model has no non-LoRA matrix weight to snapshot")

    def unchanged(self, model):
        for name, parameter in model.named_parameters():
            if name == self.name:
                now = parameter.detach().reshape(-1)[:len(self.values)].cpu()
                return bool(torch.equal(self.values, now))
        raise ValueError("snapshotted parameter no longer exists")


class OptimizerStepCounter:
    """Observe actual optimizer steps without replacing the Trainer optimizer."""
    def __init__(self, optimizer):
        self.count = 0
        underlying = getattr(optimizer, 'optimizer', optimizer)
        self._hook = underlying.register_step_post_hook(self._after_step)

    def _after_step(self, optimizer, args, kwargs):
        self.count += 1

    def close(self):
        self._hook.remove()
        return self.count
