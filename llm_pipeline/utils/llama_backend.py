"""
llama.cpp llama-server backend.

Talks to a running llama-server over HTTP; models are never loaded in-process.
Responsibilities:
  R2  every sampling parameter explicit on every request
  R3  refuse to send a prompt that would be silently truncated
  R5  optional prefix KV cache save/restore (secondary model only)
  R7  reproducibility metadata attached to every generation
  R10 parallelism advertised by the server, never assumed

The prompt/case-study/selection code is untouched: this module only receives a
finished prompt string and returns the model's answer.
"""
import hashlib
import re
import threading
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
BACKEND_CONFIG_PATH = ROOT / "config" / "backend.yaml"

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# An unterminated reasoning block means the answer never arrived (max_tokens too low).
UNCLOSED_THINK = re.compile(r"<think>(?!.*</think>)", re.DOTALL | re.IGNORECASE)


class ContextOverflowError(RuntimeError):
    """Prompt would exceed the per-slot context. Never retry: it will not get smaller."""


class BackendError(RuntimeError):
    """Transport or server-side failure. Retryable."""


def load_backend_config(path=None):
    path = Path(path) if path else BACKEND_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Backend config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_thinking(text):
    """
    Remove <think>...</think> so the reasoning cannot be mistaken for the answer.

    parse_llm_output uses re.search and takes the FIRST "[[ ## answer ## ]]" it finds.
    With thinking enabled the model often rehearses that exact marker inside its
    reasoning, which would yield the wrong number.
    """
    if text is None:
        return None
    if UNCLOSED_THINK.search(text):
        raise BackendError(
            "reasoning block was never closed: the answer was cut off, raise sampling.max_tokens"
        )
    return THINK_BLOCK.sub("", text).strip()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class LlamaServerBackend:
    """
    One instance per process. Thread-safe for concurrent generate() calls, which is
    what R10 needs on the secondary model (-np 4).
    """

    def __init__(self, config=None, model_key=None):
        self.cfg = config or load_backend_config()

        key = model_key or self.cfg.get("active_model", "primary")
        models = self.cfg.get("models", {})
        if key not in models:
            raise ValueError(f"Unknown model '{key}'. Available: {sorted(models)}")
        self.model_key = key
        self.model_cfg = models[key]

        srv = self.cfg.get("server", {})
        self.url = srv.get("url", "http://127.0.0.1:8080").rstrip("/")
        self.timeout = srv.get("timeout_s", 900)
        self.endpoint = srv.get("endpoint", "chat")

        self.sampling = dict(self.cfg.get("sampling", {}))
        self.thinking = dict(self.cfg.get("thinking", {}))
        self.guard = dict(self.cfg.get("context_guard", {}))
        self.throttle = dict(self.cfg.get("throttle", {}))

        self.prefix_cache = bool(self.model_cfg.get("prefix_cache", False))
        if self.prefix_cache and self.endpoint != "completion":
            # id_slot only exists on /completion
            self.endpoint = "completion"

        self._probe_lock = threading.Lock()
        self._slot_counter = 0
        self._counter_lock = threading.Lock()
        self._props = None
        self._gguf_sha = None
        self._context_per_slot = None
        self._total_slots = None
        self._build_info = None
        self._template_supported = None

    # ------------------------------------------------------------------ probing

    def _get(self, path):
        try:
            r = requests.get(f"{self.url}{path}", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise BackendError(f"GET {path} failed: {e}") from e

    def _post(self, path, payload):
        try:
            r = requests.post(f"{self.url}{path}", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise BackendError(f"POST {path} failed: {e}") from e

    def probe(self):
        """Read server properties once. Determines the real per-slot context (R3)."""
        with self._probe_lock:
            if self._props is not None:
                return self._props

            props = self._get("/props")
            self._props = props

            gen = props.get("default_generation_settings") or {}
            # default_generation_settings.n_ctx is authoritative and is ALREADY the
            # per-slot context. Verified against build b10653: a server started with
            # -c 16384 reports total_slots=4 and n_ctx=16384, i.e. with kv_unified the
            # context is NOT divided across slots. Dividing here would have rejected
            # almost every valid prompt as an overflow.
            per_slot = gen.get("n_ctx") or props.get("n_ctx")
            total_slots = props.get("total_slots") or gen.get("n_parallel")

            override = (self.cfg.get("server", {}) or {}).get("context_per_slot_override")
            if override:
                per_slot = int(override)

            if not per_slot:
                raise BackendError(
                    "could not determine per-slot context from /props; "
                    "set server.context_per_slot_override in config/backend.yaml"
                )

            self._context_per_slot = int(per_slot)
            self._total_slots = int(total_slots or 1)
            self._build_info = (
                props.get("build_info")
                or props.get("system_info")
                or gen.get("build_info")
                or "unknown"
            )

            if self.cfg.get("reproducibility", {}).get("hash_gguf", False):
                model_path = props.get("model_path") or gen.get("model")
                if model_path and Path(model_path).exists():
                    self._gguf_sha = sha256_file(model_path)
                else:
                    self._gguf_sha = f"unavailable (model_path={model_path!r})"

            return props

    def _ctx_from_flags(self):
        m = re.search(r"-c\s+(\d+)", self.model_cfg.get("server_flags", ""))
        return int(m.group(1)) if m else None

    @property
    def context_per_slot(self):
        if self._context_per_slot is None:
            self.probe()
        return self._context_per_slot

    @property
    def total_slots(self):
        if self._total_slots is None:
            self.probe()
        return self._total_slots

    def concurrency(self):
        """
        How many requests to keep in flight. Config value, capped by what the server
        actually advertises: the client adapts instead of assuming (R10).
        """
        want = int(self.model_cfg.get("concurrency", 1))
        return max(1, min(want, self.total_slots))

    # ----------------------------------------------------------------- tokenising

    def tokenize(self, text):
        return len(self._post("/tokenize", {"content": text}).get("tokens", []))

    def _templated_prompt(self, prompt):
        """
        Exact string the server will process, chat template included. Needed so the
        R3 guard counts what is really sent and not just the bare prompt.
        Returns None when the server has no /apply-template.
        """
        if self._template_supported is False:
            return None
        try:
            body = {"messages": [{"role": "user", "content": prompt}]}
            if self.thinking.get("enabled") is not None:
                body["chat_template_kwargs"] = {"enable_thinking": bool(self.thinking.get("enabled"))}
            out = self._post("/apply-template", body)
            self._template_supported = True
            return out.get("prompt")
        except BackendError:
            self._template_supported = False
            return None

    def check_context(self, prompt, templated=None):
        """
        R3: llama.cpp truncates silently when prompt + generation exceed the slot.
        Count first, refuse before sending.
        """
        if not self.guard.get("enabled", True):
            return None

        text = templated if templated is not None else prompt
        n_prompt = self.tokenize(text)
        if templated is None:
            n_prompt += int(self.guard.get("template_overhead_fallback", 64))

        budget = self.context_per_slot
        need = n_prompt + int(self.sampling.get("max_tokens", 0)) + int(self.guard.get("reserve_tokens", 0))

        if need > budget:
            raise ContextOverflowError(
                f"prompt {n_prompt} tok + max_tokens {self.sampling.get('max_tokens')} "
                f"+ reserve {self.guard.get('reserve_tokens', 0)} = {need} tok "
                f"exceeds the {budget} tok available per slot "
                f"(server -c / {self.total_slots} slots). Case skipped, not retried."
            )
        return n_prompt

    # ------------------------------------------------------------------- metadata

    def static_metadata(self):
        """R7: everything needed to reproduce this run, minus the per-call timings."""
        self.probe()
        return {
            "backend": "llamacpp",
            "model_key": self.model_key,
            "model_name": self.model_cfg.get("name"),
            "quant": self.model_cfg.get("quant"),
            "hf_repo": self.model_cfg.get("hf_repo"),
            "gguf_sha256": self._gguf_sha,
            "llamacpp_build": self._build_info,
            "llamacpp_build_tag": (self.cfg.get("server", {}) or {}).get("build_tag"),
            "server_flags": " ".join((self.model_cfg.get("server_flags") or "").split()),
            "server_url": self.url,
            "endpoint": self.endpoint,
            "context_per_slot": self.context_per_slot,
            "total_slots": self.total_slots,
            "sampling": dict(self.sampling),
            "thinking_enabled": bool(self.thinking.get("enabled")),
            "prefix_cache_regime": bool(self.prefix_cache),
            "throttle_sleep_factor": float(self.throttle.get("sleep_factor", 0) or 0),
        }

    # ----------------------------------------------------------------- generation

    def _next_slot(self):
        with self._counter_lock:
            slot = self._slot_counter % self.total_slots
            self._slot_counter += 1
        return slot

    def _payload_common(self):
        s = self.sampling
        return {
            "temperature": s.get("temperature"),
            "top_p": s.get("top_p"),
            "top_k": s.get("top_k"),
            "seed": s.get("seed"),
        }

    def generate(self, prompt):
        """
        Send one prompt, return (answer_text, metadata).
        Raises ContextOverflowError (do not retry) or BackendError (retryable).
        """
        self.probe()
        templated = self._templated_prompt(prompt) if self.endpoint == "completion" else None
        n_prompt = self.check_context(prompt, templated)

        started = time.perf_counter()

        if self.endpoint == "completion":
            body = dict(self._payload_common())
            body.update({
                "prompt": templated if templated is not None else prompt,
                "n_predict": self.sampling.get("max_tokens"),
                "cache_prompt": True,
            })
            if self.prefix_cache:
                body["id_slot"] = self._next_slot()
            resp = self._post("/completion", body)
            raw = resp.get("content")
            timings = resp.get("timings", {}) or {}
            reasoning = None
        else:
            body = dict(self._payload_common())
            body.update({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.sampling.get("max_tokens"),
            })
            if self.thinking.get("enabled") is not None:
                body["chat_template_kwargs"] = {"enable_thinking": bool(self.thinking.get("enabled"))}
            resp = self._post("/v1/chat/completions", body)
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message", {}) or {}
            raw = msg.get("content")
            # Recent builds split the reasoning out already; then no stripping is needed.
            reasoning = msg.get("reasoning_content")
            timings = resp.get("timings", {}) or {}
            usage = resp.get("usage", {}) or {}
            if not timings and usage:
                timings = {
                    "prompt_n": usage.get("prompt_tokens"),
                    "predicted_n": usage.get("completion_tokens"),
                }

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answer = raw
        if reasoning is None and self.thinking.get("strip_from_output", True):
            answer = strip_thinking(raw)

        # Give the GPU idle time between requests (see throttle: in backend.yaml).
        factor = float(self.throttle.get("sleep_factor", 0) or 0)
        if factor > 0:
            time.sleep(min(elapsed_ms / 1000.0 * factor,
                           float(self.throttle.get("max_sleep_s", 10))))

        meta = {
            "timestamp_ms": int(time.time() * 1000),
            "duration_ms": round(elapsed_ms, 2),
            "prompt_tokens": timings.get("prompt_n") or n_prompt,
            "completion_tokens": timings.get("predicted_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
            "prompt_per_second": timings.get("prompt_per_second"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "reasoning_returned_separately": reasoning is not None,
            "reasoning_chars": len(reasoning) if reasoning else (
                len(raw) - len(answer) if raw and answer else 0
            ),
        }
        return answer, meta

    # ------------------------------------------------------------ prefix KV cache

    def build_prefix_cache(self, prefix_text, filename, slot=0):
        """R5: process the shared prefix once and persist its KV cache."""
        body = dict(self._payload_common())
        body.update({"prompt": prefix_text, "n_predict": 1, "id_slot": slot, "cache_prompt": True})
        self._post("/completion", body)
        self._post(f"/slots/{slot}?action=save", {"filename": filename})
        return filename

    def restore_prefix_cache(self, filename, slot=0):
        self._post(f"/slots/{slot}?action=restore", {"filename": filename})

    # ------------------------------------------------------------------- selftest

    def selftest(self):
        """
        Report which server capabilities are actually available on this build.
        Written because these endpoints vary between llama.cpp versions and the
        integration was authored without a running server.
        """
        report = {}
        try:
            self.probe()
            report["props"] = "ok"
            report["context_per_slot"] = self.context_per_slot
            report["total_slots"] = self.total_slots
            report["build"] = self._build_info
            report["gguf_sha256"] = self._gguf_sha
        except Exception as e:
            report["props"] = f"FAIL: {e}"
            return report

        try:
            report["tokenize"] = f"ok ({self.tokenize('hello world')} tok)"
        except Exception as e:
            report["tokenize"] = f"FAIL: {e}"

        t = self._templated_prompt("hello")
        report["apply_template"] = "ok" if t else "unavailable (falling back to raw + overhead)"

        try:
            ans, meta = self.generate("Reply with the single digit 7 and nothing else.")
            report["generate"] = f"ok -> {ans!r}"
            report["timings_available"] = bool(meta.get("predicted_per_second"))
            report["reasoning_split"] = meta.get("reasoning_returned_separately")
        except Exception as e:
            report["generate"] = f"FAIL: {type(e).__name__}: {e}"

        return report


# One backend per model key for the whole process. Probing costs a /props round trip
# plus a full SHA256 of a ~5.5 GB GGUF, so concurrent workers must share the instance
# instead of each building their own.
_BACKEND_CACHE = {}
_BACKEND_CACHE_LOCK = threading.Lock()


def get_shared_backend(model_key=None):
    cfg = load_backend_config()
    key = model_key or cfg.get("active_model", "primary")
    with _BACKEND_CACHE_LOCK:
        if key not in _BACKEND_CACHE:
            _BACKEND_CACHE[key] = LlamaServerBackend(config=cfg, model_key=key)
        return _BACKEND_CACHE[key]


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="llama-server backend selftest")
    ap.add_argument("--model", default=None, help="model key from backend.yaml (primary/secondary)")
    args = ap.parse_args()

    be = LlamaServerBackend(model_key=args.model)
    print(json.dumps(be.selftest(), indent=2, ensure_ascii=False))
