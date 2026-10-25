"""Small, safe helpers used by the prompt-engineering examples.

The module deliberately does not contact a model service at import or client
construction time.  Calls to local Ollama and optional OpenAI are explicit.
"""
from __future__ import annotations

import ast
import base64
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROFILES = {
    "short": {"max_tokens": 192, "image_steps": 8, "vote_samples": 3},
    "full": {"max_tokens": 256, "image_steps": 20, "vote_samples": 5},
}
IMAGE_MODEL = "prompthero/openjourney"
IMAGE_REVISION = "f4572661b028c732b2b97c8fbdc32fa5db3afe03"


class PromptClient:
    """A minimal client which uses Ollama's native API for local models."""

    def __init__(self, provider: str = "ollama", model: str = "gemma4:31b-mlx",
                 vision_model: str | None = None, base_url: str = "http://localhost:11434",
                 timeout: int = 120, max_tokens: int = 192, seed: int = 42):
        if provider not in {"ollama", "openai"}:
            raise ValueError("provider must be 'ollama' or 'openai'")
        if not model or not isinstance(model, str):
            raise ValueError("a model must be configured explicitly")
        self.provider, self.model, self.vision_model = provider, model, vision_model
        self.base_url, self.timeout = base_url, timeout
        self.max_tokens, self.seed = max_tokens, seed
        self.records: list[dict[str, Any]] = []
        # SDK clients are intentionally lazy: construction must be offline.
        self._client = None

    @staticmethod
    def _name_installed(configured: str, installed: set[str]) -> bool:
        return configured in installed or (":" not in configured and configured + ":latest" in installed)

    def preflight(self) -> dict[str, Any]:
        result: dict[str, Any] = {"provider": self.provider, "model": self.model,
                                  "vision_model": self.vision_model}
        if self.provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key or self._placeholder_key(key):
                raise RuntimeError("OpenAI was selected but OPENAI_API_KEY is not configured")
            result["capabilities"] = "not network-tested (cloud preflight is credential-only)"
            return result
        try:
            import requests
            response = requests.get(self.base_url.rstrip("/") + "/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("Cannot reach local Ollama at configured base_url; start Ollama and check it is running") from exc
        models = payload.get("models", [])
        names = {item.get("name") or item.get("model") for item in models
                 if isinstance(item, dict) and (item.get("name") or item.get("model"))}
        missing = [name for name in (self.model, self.vision_model) if name and not self._name_installed(name, names)]
        if missing:
            raise RuntimeError("Configured Ollama model(s) not installed: " + ", ".join(missing) + ". Pull the named model explicitly.")
        result["installed_models"] = sorted(names)
        capabilities = {item.get("name") or item.get("model"): item.get("capabilities", "unknown")
                        for item in models if isinstance(item, dict)}
        result["capabilities"] = capabilities
        if self.vision_model:
            declared = capabilities.get(self.vision_model, capabilities.get(self.vision_model + ":latest", "unknown"))
            if isinstance(declared, list) and "vision" not in declared:
                raise RuntimeError('Configured vision model does not declare vision support; select a vision model or disable RUN_VISION')
            result["vision_capabilities"] = declared
        return result

    @staticmethod
    def _placeholder_key(key: str) -> bool:
        lowered = key.strip().lower()
        return lowered in {"", "none", "null", "ollama"} or "your" in lowered or "placeholder" in lowered or lowered.startswith("<")

    def _ollama(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
            except Exception as exc:
                raise RuntimeError("Unable to initialize local Ollama SDK") from exc
        return self._client

    def _openai(self):
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key or self._placeholder_key(key):
            raise RuntimeError("OpenAI was selected but OPENAI_API_KEY is not configured")
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=key, timeout=self.timeout)
            except Exception as exc:
                raise RuntimeError("Unable to initialize OpenAI client") from exc
        return self._client

    @staticmethod
    def _image_name(image: Any) -> str:
        return os.path.basename(os.fspath(image)) if isinstance(image, (str, os.PathLike)) else "image-bytes"

    @staticmethod
    def _cloud_image(image: Any) -> str:
        if isinstance(image, (str, os.PathLike)):
            path = Path(image)
            data = path.read_bytes()
        elif isinstance(image, bytes):
            data = image
        else:
            raise TypeError("images must be paths or bytes")
        # Inspect bytes: the historical .jpg OCR fixture actually contains PNG data.
        from PIL import Image
        with Image.open(io.BytesIO(data)) as decoded:
            mime = Image.MIME.get(decoded.format)
        if mime not in {'image/png', 'image/jpeg', 'image/webp', 'image/gif'}:
            raise ValueError('Unsupported image format')
        return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0,
                 images: list[Any] | None = None, max_tokens: int | None = None,
                 seed: int | None = None) -> str:
        if images is not None and not isinstance(images, list):
            raise TypeError("images must be a list or None")
        budget, actual_seed = max_tokens if max_tokens is not None else self.max_tokens, seed if seed is not None else self.seed
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError('max_tokens must be a positive integer')
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be a finite value between 0 and 2')
        model = self.vision_model if images and self.vision_model else self.model
        started = time.monotonic()
        finish_reason = None
        try:
            if self.provider == "ollama":
                messages: list[dict[str, Any]] = []
                if system:
                    messages.append({"role": "system", "content": system})
                user: dict[str, Any] = {"role": "user", "content": prompt}
                if images:
                    user["images"] = images
                messages.append(user)
                response = self._ollama().chat(model=model, messages=messages, stream=False, think=False,
                                               options={"temperature": temperature, "num_predict": budget, "seed": actual_seed})
                message = response.get("message", {}) if isinstance(response, dict) else response.message
                content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
                finish_reason = response.get("done_reason") if isinstance(response, dict) else getattr(response, "done_reason", None)
            else:
                content: Any = prompt
                if images:
                    content = [{"type": "text", "text": prompt}] + [
                        {"type": "image_url", "image_url": {"url": self._cloud_image(image)}} for image in images]
                messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
                response = self._openai().chat.completions.create(model=model, messages=messages,
                    max_completion_tokens=budget, temperature=temperature, seed=actual_seed)
                choice = response.choices[0]
                content, finish_reason = choice.message.content, choice.finish_reason
        except RuntimeError:
            raise
        except Exception:
            # Do not surface SDK request details: they can contain headers or file paths.
            raise RuntimeError("Model request failed; verify provider, configured model, and local service") from None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Model returned empty content; increase completion budget or disable reasoning/thinking for this request")
        self.records.append({"index": len(self.records), "provider": self.provider, "model": model,
            "prompt": prompt, "system": system, "image_count": len(images or []),
            "image_names": [self._image_name(x) for x in images or []], "temperature": temperature,
            "seed": actual_seed, "max_tokens": budget, "elapsed_seconds": time.monotonic() - started,
            "response": content, "finish_reason": finish_reason})
        return content


def parse_json_object(text: str, required_keys: list[str] | None = None) -> dict[str, Any]:
    """Parse only a literal JSON object; markdown fences are intentionally rejected."""
    def reject_constant(value):
        raise ValueError('Nonfinite constants are not valid JSON')
    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("expected a strict JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    if required_keys and not set(required_keys).issubset(value):
        raise ValueError("JSON object is missing required keys")
    return value


def majority_vote(values: list[Any]) -> dict[str, Any]:
    valid = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    if not valid:
        return {"winner": None, "count": 0, "valid_samples": 0, "tie": False}
    counts = Counter(valid)
    highest = max(counts.values())
    winners = [value for value, count in counts.items() if count == highest]
    return {"winner": winners[0] if len(winners) == 1 else None, "count": highest,
            "valid_samples": len(valid), "tie": len(winners) > 1}


def safe_calculate(expression: str) -> int | float:
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("expression is too long or invalid")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise ValueError("expression is too complex")
    def check(value: Any) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1_000_000:
            raise ValueError("non-finite or out-of-range arithmetic value")
        return value
    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant): return check(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return check((1 if isinstance(node.op, ast.UAdd) else -1) * visit(node.operand))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): result = left + right
            elif isinstance(node.op, ast.Sub): result = left - right
            elif isinstance(node.op, ast.Mult): result = left * right
            else:
                if right == 0: raise ValueError("division by zero")
                result = left / right
            return check(result)
        raise ValueError("only +, -, *, /, parentheses, and numeric literals are allowed")
    return visit(tree.body)


def run_react(client: Any, question: str, max_turns: int = 3) -> dict[str, Any]:
    if max_turns < 1: raise ValueError("max_turns must be positive")
    trace, used_calculator, observations = [], False, []
    system = ("Return strict JSON only. Actions are calculate(expression) encoded as "
              '{"action":"calculate","expression":"..."} or final(answer) encoded as '
              '{"action":"final","answer":...}. Use calculate for arithmetic; do not provide private reasoning.')
    for turn in range(1, max_turns + 1):
        prompt = "Question: " + question + "\nStructured observations: " + json.dumps(observations)
        data = parse_json_object(client.complete(prompt, system=system, temperature=0))
        action = data.get("action")
        if action == "calculate":
            if not isinstance(data.get("expression"), str): raise ValueError("calculate action requires a string expression")
            result = safe_calculate(data["expression"])
            used_calculator = True
            item = {"turn": turn, "action": "calculate", "expression": data["expression"], "observation": result}
            trace.append(item); observations.append(item)
        elif action == "final":
            if not used_calculator:
                item = {"turn": turn, "action": "final", "observation": "final refused: calculate first"}
                trace.append(item); observations.append(item)
                continue
            answer = data.get("answer")
            if not isinstance(answer, (str, int, float)) or isinstance(answer, bool): raise ValueError("final action requires a string or number answer")
            trace.append({"turn": turn, "action": "final", "observation": answer})
            return {"answer": answer, "trace": trace, "used_calculator": True}
        else: raise ValueError("unknown or malformed ReAct action")
    raise ValueError("ReAct exhausted its turn budget without a valid final answer")


def choose_image_device(choice: str = "auto") -> str:
    import torch
    available = {"cuda": bool(torch.cuda.is_available()), "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()), "cpu": True}
    if choice == "auto": return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if choice not in available: raise ValueError("device must be auto, cuda, mps, or cpu")
    if not available[choice]: raise RuntimeError("requested image device is unavailable: " + choice)
    return choice


def load_image_pipeline(device_choice: str = "auto") -> tuple[Any, str]:
    import torch
    from diffusers import DiffusionPipeline
    device = choose_image_device(device_choice)
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    # Current Diffusers uses dtype; this is a built-in pipeline, not remote code.
    pipe = DiffusionPipeline.from_pretrained(IMAGE_MODEL, revision=IMAGE_REVISION, dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe, device


def generate_image(pipe: Any, prompt: str, negative_prompt: str = "", seed: int = 42,
                   steps: int = 8, width: int = 512, height: int = 512) -> Any:
    import torch
    for name, value in (("steps", steps), ("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(name + " must be a positive integer")
    if not 1 <= steps <= 50 or width < 128 or height < 128 or width > 768 or height > 768 or width % 8 or height % 8:
        raise ValueError("steps and dimensions exceed safe image-generation bounds")
    generator = torch.Generator("cpu").manual_seed(seed)
    output = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator,
                  num_inference_steps=steps, guidance_scale=7.5, width=width, height=height)
    return output.images[0]
