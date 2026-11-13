"""
Z AI (GLM) backend.

Same contract as the llama.cpp backend: it receives an already-built prompt and returns
the model's answer, so the prompt / case-study / selection code is untouched and the two
experiments run the exact same prompt text.

Responsibilities mirrored from the local backend:
  R2  every sampling parameter explicit on every request
  R3  refuse to send a prompt that would exceed the context window
  R7  reproducibility metadata attached to every generation
  R10 concurrency is a configuration value

Additionally, and unlike the local backend, every request costs money, so token usage is
priced per call and accumulated into a running total that the pilot reports.

Note on thinking: GLM-5.3 and GLM-5.3-Flash do NOT support disabling it. The API returns
an error if thinking.type is set to "disabled", so the reasoning tokens are unavoidable
and are billed at the output rate.
"""
import os
import re
import threading
import time
from pathlib import Path

import requests
import yaml

from utils.llama_backend import ContextOverflowError, BackendError, strip_thinking


class FatalApiError(RuntimeError):
    """
    An error that no amount of retrying can fix: bad key, no credit, model not enabled.

    Kept separate from BackendError because the two need opposite handling. A timeout
    should back off and retry; an empty wallet should stop the run immediately. Z AI
    reports insufficient balance as HTTP 429, the same status as a genuine rate limit,
    so the distinction has to be made on the error code in the body rather than on the
    status alone.
    """


# Business codes that mean "stop", returned alongside a 429 that otherwise looks retryable.
FATAL_ERROR_CODES = {
    "1113",   # Insufficient balance or no resource package
    "1112",   # Account suspended / arrears
    "1002",   # Invalid API key
    "1004",   # Authentication failed
}

ROOT = Path(__file__).resolve().parents[1]
BACKEND_CONFIG_PATH = ROOT / "config" / "backend.yaml"


def load_backend_config(path=None):
    path = Path(path) if path else BACKEND_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Backend config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ZaiBackend:
    """One instance per process, shared across worker threads."""

    def __init__(self, config=None, model_key=None, api_key=None):
        self.cfg = config or load_backend_config()
        zcfg = self.cfg.get("zai", {}) or {}

        key = model_key or zcfg.get("active_model", "glm-5.3-flash")
        models = zcfg.get("models", {})
        if key not in models:
            raise ValueError(f"Unknown Z AI model '{key}'. Available: {sorted(models)}")
        self.model_key = key
        self.model_cfg = models[key]

        self.url = zcfg.get("url", "https://api.z.ai/api/paas/v4/chat/completions")
        self.timeout = zcfg.get("timeout_s", 900)
        self.sampling = dict(zcfg.get("sampling", {}))
        self.thinking = dict(zcfg.get("thinking", {}))
        self.guard = dict(zcfg.get("context_guard", {}))
        self.cache_cfg = dict(zcfg.get("cache", {}))

        # An explicitly supplied key wins over the environment. It is never written to
        # the metadata, the results records or any log line.
        self._api_key = api_key
        self._lock = threading.Lock()
        self._spend = {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0,
                       "output_tokens": 0, "usd": 0.0, "usd_without_cache": 0.0}

    # ---------------------------------------------------------------------- auth

    @property
    def api_key(self):
        """
        Read lazily so that importing the module, or running any other provider, does
        not require the key to be present.
        """
        if self._api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            self._api_key = os.getenv("ZAI_API_KEY")
            if not self._api_key:
                raise BackendError("ZAI_API_KEY not set: add it to the .env file in the project root")
        return self._api_key

    # ------------------------------------------------------------------ counting

    def estimate_tokens(self, text):
        """
        Token estimate. The Z AI API exposes no tokenizer endpoint, so this is a
        character-based approximation, deliberately conservative. It is only used by the
        pre-send guard; the authoritative counts come back in the response usage block
        and are what gets billed and recorded.
        """
        return int(len(text) / float(self.guard.get("chars_per_token", 3.2))) + 1

    def check_context(self, prompt):
        """R3: refuse a prompt that cannot fit, instead of letting the API truncate it."""
        if not self.guard.get("enabled", True):
            return None
        n = self.estimate_tokens(prompt)
        budget = int(self.model_cfg.get("context_window", 0))
        need = n + int(self.sampling.get("max_tokens", 0)) + int(self.guard.get("reserve_tokens", 0))
        if budget and need > budget:
            raise ContextOverflowError(
                f"prompt ~{n} tok + max_tokens {self.sampling.get('max_tokens')} = {need} tok "
                f"exceeds the {budget} tok context of {self.model_cfg.get('name')}. "
                f"Case skipped, not retried."
            )
        return n

    def request_sampling(self):
        """
        The sampling fields actually sent, filtered to what this model documents.

        R2 wants every parameter explicit, but sending an undocumented one to a hosted API
        risks a 400 rather than being ignored. Each model therefore declares what it
        accepts, and anything else is dropped here instead of being silently assumed.
        thinking.type is nested rather than flat, and is never sent as "disabled" on a
        model that rejects it.
        """
        s = self.sampling
        allowed = set(self.model_cfg.get("supported_sampling") or s.keys())
        out = {}

        for name in ("temperature", "top_p", "max_tokens", "seed", "reasoning_effort"):
            if name in allowed and s.get(name) is not None:
                out[name] = s[name]

        if "thinking_type" in allowed:
            want = s.get("thinking_type", "enabled")
            if want == "disabled" and not self.model_cfg.get("thinking_disableable", False):
                raise BackendError(
                    f"{self.model_cfg.get('name')} does not support disabling thinking: "
                    f"the API rejects thinking.type='disabled'. Use glm-5.2 instead."
                )
            thinking = {"type": want}
            if "clear_thinking" in allowed and s.get("clear_thinking") is not None:
                thinking["clear_thinking"] = bool(s["clear_thinking"])
            out["thinking"] = thinking
        return out

    # ------------------------------------------------------------------- spending

    @staticmethod
    def _dig(usage, dotted):
        """Read a possibly nested usage field such as prompt_tokens_details.cached_tokens."""
        cur = usage
        for part in dotted.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur if isinstance(cur, int) else None

    def cached_tokens(self, usage):
        """
        How many prompt tokens the API served from its context cache. The field name is
        not documented, so the candidates from config are tried in order; None means the
        response carried no hit count and everything is billed at the full input rate.
        """
        if not self.cache_cfg.get("account_for_hits", True):
            return None
        for field in self.cache_cfg.get("usage_fields", []):
            v = self._dig(usage, field)
            if v is not None:
                return v
        return None

    def _bill(self, in_tok, out_tok, cached_tok=None):
        """
        Cached prompt tokens are billed at the "Cached Input" rate, roughly 80% off.
        The uncached cost is tracked alongside so the saving is measurable rather than
        assumed: the two figures together say what the cache is actually worth here.
        """
        pin = float(self.model_cfg.get("price_input_per_mtok", 0.0))
        pcached = float(self.model_cfg.get("price_cached_input_per_mtok", pin))
        pout = float(self.model_cfg.get("price_output_per_mtok", 0.0))

        hit = min(cached_tok or 0, in_tok)
        miss = in_tok - hit
        usd = (miss / 1e6 * pin) + (hit / 1e6 * pcached) + (out_tok / 1e6 * pout)
        usd_nocache = (in_tok / 1e6 * pin) + (out_tok / 1e6 * pout)

        with self._lock:
            self._spend["requests"] += 1
            self._spend["input_tokens"] += in_tok
            self._spend["cached_input_tokens"] += hit
            self._spend["output_tokens"] += out_tok
            self._spend["usd"] += usd
            self._spend["usd_without_cache"] += usd_nocache
        return usd, hit

    def spend(self):
        with self._lock:
            return dict(self._spend)

    # ------------------------------------------------------------------- metadata

    def static_metadata(self):
        """R7: what is needed to reproduce this run."""
        return {
            "backend": "zai",
            "model_key": self.model_key,
            "model_name": self.model_cfg.get("name"),
            "api_model_id": self.model_cfg.get("api_id"),
            "endpoint": self.url,
            "context_window": self.model_cfg.get("context_window"),
            "sampling": dict(self.sampling),
            "sampling_sent": self.request_sampling(),
            "thinking_enabled": bool(self.thinking.get("enabled", True)),
            "thinking_disableable": bool(self.model_cfg.get("thinking_disableable", False)),
            "price_input_per_mtok": self.model_cfg.get("price_input_per_mtok"),
            "price_cached_input_per_mtok": self.model_cfg.get("price_cached_input_per_mtok"),
            "price_output_per_mtok": self.model_cfg.get("price_output_per_mtok"),
        }

    # ----------------------------------------------------------------- generation

    def generate(self, prompt):
        """
        Send one prompt, return (answer_text, metadata).
        Raises ContextOverflowError (never retry) or BackendError (retryable).
        """
        est = self.check_context(prompt)

        body = {
            "model": self.model_cfg.get("api_id"),
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(self.request_sampling())

        started = time.perf_counter()
        try:
            r = requests.post(
                self.url, json=body, timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
        except Exception as e:
            raise BackendError(f"POST {self.url} failed: {e}") from e

        if r.status_code != 200:
            code, message = None, r.text[:300]
            try:
                err = (r.json() or {}).get("error") or {}
                code, message = str(err.get("code", "")), err.get("message", message)
            except Exception:
                pass

            # 401/403 are always fatal; 429 is fatal only when the body says the problem
            # is the balance rather than the rate.
            if r.status_code in (401, 403) or code in FATAL_ERROR_CODES:
                raise FatalApiError(
                    f"Z AI refused the request and retrying cannot help "
                    f"(HTTP {r.status_code}, code {code or 'n/a'}): {message}"
                )
            raise BackendError(f"HTTP {r.status_code} code={code or 'n/a'}: {message}")
        try:
            resp = r.json()
        except Exception as e:
            raise BackendError(f"malformed JSON response: {e}") from e

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        raw = msg.get("content")
        reasoning = msg.get("reasoning_content")

        # When the API keeps reasoning separate, content is already clean; otherwise the
        # <think> block is stripped so parse_llm_output cannot pick up a number that the
        # model merely rehearsed while reasoning.
        answer = raw if reasoning is not None else strip_thinking(raw)

        usage = resp.get("usage", {}) or {}
        in_tok = usage.get("prompt_tokens") or est or 0
        out_tok = usage.get("completion_tokens") or 0
        # Reasoning tokens are billed as output; the API may or may not break them out.
        reasoning_tok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        cached_tok = self.cached_tokens(usage)
        usd, hit = self._bill(in_tok, out_tok, cached_tok)

        meta = {
            "timestamp_ms": int(time.time() * 1000),
            "duration_ms": round(elapsed_ms, 2),
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "reasoning_tokens": reasoning_tok,
            "finish_reason": choice.get("finish_reason"),
            "reasoning_returned_separately": reasoning is not None,
            "reasoning_chars": len(reasoning) if reasoning else (
                len(raw) - len(answer) if raw and answer else 0),
            "cached_input_tokens": hit,
            "cache_hit_reported": cached_tok is not None,
            "cost_usd": round(usd, 6),
            "cumulative_cost_usd": round(self.spend()["usd"], 4),
        }

        # A run that hits the cap has spent its whole budget on reasoning and never
        # emitted the answer. This is exactly how the local Qwen run failed, so it is
        # surfaced as an error rather than left to fail silently in the parser.
        if choice.get("finish_reason") == "length":
            raise BackendError(
                f"generation stopped at max_tokens ({s.get('max_tokens')}) before producing an "
                f"answer: the model was still reasoning. Raise zai.sampling.max_tokens."
            )
        return answer, meta

    # ------------------------------------------------------------------- selftest

    def selftest(self):
        report = {"model": self.model_cfg.get("name"), "endpoint": self.url}
        try:
            report["api_key"] = "present" if self.api_key else "missing"
        except BackendError as e:
            report["api_key"] = f"FAIL: {e}"
            return report
        try:
            ans, meta = self.generate("Reply with the single digit 7 and nothing else.")
            report["generate"] = f"ok -> {ans!r}"
            report["prompt_tokens"] = meta["prompt_tokens"]
            report["completion_tokens"] = meta["completion_tokens"]
            report["reasoning_tokens"] = meta["reasoning_tokens"]
            report["reasoning_split"] = meta["reasoning_returned_separately"]
            report["cost_usd"] = meta["cost_usd"]
            report["cached_input_tokens"] = meta["cached_input_tokens"]
            report["cache_hit_reported"] = meta["cache_hit_reported"]
            report["sampling_sent"] = self.request_sampling()
        except Exception as e:
            report["generate"] = f"FAIL: {type(e).__name__}: {e}"
        return report


_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_shared_zai_backend(model_key=None, api_key=None):
    cfg = load_backend_config()
    key = model_key or (cfg.get("zai", {}) or {}).get("active_model", "glm-5.3-flash")
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = ZaiBackend(config=cfg, model_key=key, api_key=api_key)
        elif api_key:
            _CACHE[key]._api_key = api_key
        return _CACHE[key]


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Z AI backend selftest")
    ap.add_argument("--model", default=None, help="model key from backend.yaml zai.models")
    ap.add_argument("--api-key", nargs="?", const="__ASK__", default=None,
                    help="API key: a value, '-' for stdin, or the flag alone to be prompted")
    args = ap.parse_args()

    from utils.general_utils import resolve_api_key
    backend = get_shared_zai_backend(args.model, resolve_api_key(args.api_key))
    print(json.dumps(backend.selftest(), indent=2, ensure_ascii=False))
