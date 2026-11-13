"""
Anthropic (Claude) backend.

Same contract as the Z AI and llama.cpp backends: it receives an already-built prompt
and returns the model's answer, so the prompt / case-study / selection code is
untouched and the four models of the comparison run the exact same prompt text.

Responsibilities mirrored from the other backends:
  R2  every generation parameter explicit on every request
  R3  refuse to send a prompt that would exceed the context window
  R7  reproducibility metadata attached to every generation
  R10 concurrency is a configuration value

Two execution paths share one request builder, so a prompt is encoded identically
whichever way it is sent:
  * synchronous  -> generate(prompt)            used by test_llm.py (latency sample)
  * batch        -> request_params(prompt)       used by run_anthropic_batch.py, which
                                                  submits through the Message Batches API
                                                  at half price and calls parse_message()
                                                  on each result.

Notes on Claude Sonnet 5.5 that shaped this file:
  * thinking.type "disabled" is rejected; "between_tools" is the lowest setting and,
    with no tools declared, means no extended thinking at all.
  * non-default temperature / top_p / top_k are rejected, so none is sent.
  * a safety decline is an HTTP 200 with stop_reason "refusal": it is surfaced as a
    RefusalError and recorded as a skipped case, never retried and never routed to
    another model (a server-side fallback would contaminate a model comparison).
"""
import os
import threading
import time
from pathlib import Path

import yaml

from utils.llama_backend import ContextOverflowError, BackendError
# One FatalApiError class for every hosted provider: bad key, no credit, malformed
# request -- retrying cannot help, the whole run stops.
from utils.zai_backend import FatalApiError

ROOT = Path(__file__).resolve().parents[1]
BACKEND_CONFIG_PATH = ROOT / "config" / "backend.yaml"


class RefusalError(RuntimeError):
    """The model declined the request (stop_reason 'refusal'). Skip the case, never retry."""


class MaxTokensError(BackendError):
    """The generation hit max_tokens before the answer markers: the model reasoned in the
    text against the instructions. A property of the prompt, so the case is skipped and
    never retried (a retry costs the same prompt again for the same outcome)."""


def load_backend_config(path=None):
    path = Path(path) if path else BACKEND_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Backend config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AnthropicBackend:
    """One instance per process, shared across worker threads."""

    def __init__(self, config=None, model_key=None, api_key=None):
        self.cfg = config or load_backend_config()
        acfg = self.cfg.get("anthropic", {}) or {}

        key = model_key or acfg.get("active_model", "claude-sonnet-5-5")
        models = acfg.get("models", {})
        if key not in models:
            raise ValueError(f"Unknown Anthropic model '{key}'. Available: {sorted(models)}")
        self.model_key = key
        self.model_cfg = models[key]

        self.timeout = float(acfg.get("timeout_s", 600))
        self.sampling = dict(acfg.get("sampling", {}))
        self.thinking = dict(acfg.get("thinking", {}))
        self.guard = dict(acfg.get("context_guard", {}))
        self.cache_cfg = dict(acfg.get("cache", {}))
        self.batch_cfg = dict(acfg.get("batch", {}))
        self.concurrency = int(acfg.get("concurrency", 1))

        # An explicitly supplied key wins over the environment. It is never written to
        # the metadata, the results records or any log line.
        self._api_key = api_key
        self._client = None
        self._lock = threading.Lock()
        self._spend = {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0,
                       "cache_write_tokens": 0, "output_tokens": 0,
                       "usd": 0.0, "usd_without_cache": 0.0}

    # ---------------------------------------------------------------------- auth

    @property
    def api_key(self):
        """Read lazily so that importing the module, or running any other provider,
        does not require the key to be present."""
        if self._api_key is None:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            self._api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self._api_key:
                raise BackendError("ANTHROPIC_API_KEY not set: add it to the .env file in the project root")
        return self._api_key

    @property
    def client(self):
        if self._client is None:
            import anthropic
            # The SDK retries 429 / 5xx / connection errors twice on its own; the
            # campaign-level backoff in send_with_retry sits on top of that.
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout, max_retries=2)
        return self._client

    # ------------------------------------------------------------------ counting

    def estimate_tokens(self, text):
        """Character-based estimate for the pre-send guard; authoritative counts come
        back in the response usage block (or from count_tokens, which is free)."""
        return int(len(text) / float(self.guard.get("chars_per_token", 3.2))) + 1

    def check_context(self, prompt):
        """R3: refuse a prompt that cannot fit, instead of letting the API reject it."""
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

    def count_tokens(self, prompt):
        """Exact count through the token-counting endpoint (not billed)."""
        p = self.request_params(prompt)
        p.pop("max_tokens", None)
        p.pop("output_config", None)
        return int(self.client.messages.count_tokens(**p).input_tokens)

    # ------------------------------------------------------------------- request

    def content_blocks(self, prompt):
        """
        The prompt as user-message content blocks. When the cache breakpoint is on,
        the fixed per-log header becomes its own block carrying cache_control; the
        concatenation of the blocks is byte-identical to the prompt.
        """
        marker = self.cache_cfg.get("prefix_marker")
        if self.cache_cfg.get("prefix_breakpoint", False) and marker and marker in prompt:
            cut = prompt.index(marker)
            return [
                {"type": "text", "text": prompt[:cut], "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt[cut:]},
            ]
        return [{"type": "text", "text": prompt}]

    def thinking_params(self):
        """R2: the thinking configuration, as sent."""
        mode = self.thinking.get("mode", "between_tools")
        if mode == "between_tools":
            return {"thinking": {"type": "between_tools"}}
        if mode == "adaptive":
            return {"thinking": {"type": "adaptive"}}
        raise BackendError(f"anthropic.thinking.mode must be between_tools or adaptive, got {mode!r}")

    def request_params(self, prompt):
        """The full Messages API request for one prompt; shared by both execution paths."""
        p = {
            "model": self.model_cfg.get("api_id"),
            "max_tokens": int(self.sampling.get("max_tokens", 256)),
            "messages": [{"role": "user", "content": self.content_blocks(prompt)}],
        }
        effort = self.sampling.get("effort")
        if effort:
            p["output_config"] = {"effort": effort}
        p.update(self.thinking_params())
        return p

    # ------------------------------------------------------------------- spending

    def _bill(self, in_tok, out_tok, cache_read=0, cache_write=0, batch=False):
        """
        Four token classes, each at its own rate; the Batches API halves all of them.
        The uncached cost is tracked alongside so the saving is measurable.
        """
        disc = float(self.model_cfg.get("batch_discount", 0.5)) if batch else 1.0
        pin = float(self.model_cfg.get("price_input_per_mtok", 0.0)) * disc
        pread = float(self.model_cfg.get("price_cached_input_per_mtok", 0.0)) * disc
        pwrite = float(self.model_cfg.get("price_cache_write_per_mtok", 0.0)) * disc
        pout = float(self.model_cfg.get("price_output_per_mtok", 0.0)) * disc

        usd = (in_tok / 1e6 * pin) + (cache_read / 1e6 * pread) + (cache_write / 1e6 * pwrite) + (out_tok / 1e6 * pout)
        total_in = in_tok + cache_read + cache_write
        usd_nocache = (total_in / 1e6 * pin) + (out_tok / 1e6 * pout)

        with self._lock:
            self._spend["requests"] += 1
            self._spend["input_tokens"] += total_in
            self._spend["cached_input_tokens"] += cache_read
            self._spend["cache_write_tokens"] += cache_write
            self._spend["output_tokens"] += out_tok
            self._spend["usd"] += usd
            self._spend["usd_without_cache"] += usd_nocache
        return usd

    def spend(self):
        with self._lock:
            return dict(self._spend)

    # ------------------------------------------------------------------- metadata

    def static_metadata(self):
        """R7: what is needed to reproduce this run."""
        return {
            "backend": "anthropic",
            "model_key": self.model_key,
            "model_name": self.model_cfg.get("name"),
            "api_model_id": self.model_cfg.get("api_id"),
            "endpoint": "https://api.anthropic.com/v1/messages",
            "context_window": self.model_cfg.get("context_window"),
            "sampling": dict(self.sampling),
            "sampling_sent": {"max_tokens": self.sampling.get("max_tokens"),
                              "output_config": {"effort": self.sampling.get("effort")},
                              **self.thinking_params()},
            "sampling_configurable": bool(self.model_cfg.get("sampling_configurable", False)),
            "thinking_enabled": self.thinking.get("mode", "between_tools") != "between_tools",
            "thinking_mode": self.thinking.get("mode", "between_tools"),
            "prefix_cache": bool(self.cache_cfg.get("prefix_breakpoint", False)),
            "price_input_per_mtok": self.model_cfg.get("price_input_per_mtok"),
            "price_cached_input_per_mtok": self.model_cfg.get("price_cached_input_per_mtok"),
            "price_cache_write_per_mtok": self.model_cfg.get("price_cache_write_per_mtok"),
            "price_output_per_mtok": self.model_cfg.get("price_output_per_mtok"),
            "batch_discount": self.model_cfg.get("batch_discount"),
        }

    # ------------------------------------------------------------------- response

    def parse_message(self, msg, batch=False, elapsed_ms=None):
        """
        Turn one Message object into (answer_text, metadata), billing it on the way.
        Used by generate() and by the batch collector, so both paths score and price
        a response identically.
        """
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            expl = getattr(getattr(msg, "stop_details", None), "explanation", None)
            raise RefusalError(f"model declined the request (category={cat}): {expl}")

        # Read by block type, never by position: a thinking block may come first.
        answer = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        thinking_chars = sum(len(getattr(b, "thinking", "") or "") for b in msg.content
                             if getattr(b, "type", None) == "thinking")

        u = msg.usage
        in_tok = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        usd = self._bill(in_tok, out_tok, cache_read, cache_write, batch=batch)

        meta = {
            "timestamp_ms": int(time.time() * 1000),
            "duration_ms": round(elapsed_ms, 2) if elapsed_ms is not None else None,
            "execution": "batch" if batch else "sync",
            "prompt_tokens": in_tok + cache_read + cache_write,
            "uncached_input_tokens": in_tok,
            "cached_input_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "completion_tokens": out_tok,
            "thinking_chars": thinking_chars,
            "finish_reason": msg.stop_reason,
            "response_model": getattr(msg, "model", None),
            "request_id": getattr(msg, "_request_id", None),
            "cost_usd": round(usd, 6),
            "cumulative_cost_usd": round(self.spend()["usd"], 4),
        }

        if msg.stop_reason == "max_tokens":
            raise MaxTokensError(
                f"generation stopped at max_tokens ({self.sampling.get('max_tokens')}) before "
                f"producing an answer ({out_tok} output tokens); case skipped"
            )
        return answer, meta

    # ----------------------------------------------------------------- generation

    def generate(self, prompt):
        """
        Send one prompt synchronously, return (answer_text, metadata).
        Raises ContextOverflowError / RefusalError / MaxTokensError (never retry),
        FatalApiError (stop the run) or BackendError (retryable).
        """
        import anthropic

        self.check_context(prompt)
        params = self.request_params(prompt)

        started = time.perf_counter()
        try:
            msg = self.client.messages.create(**params)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise FatalApiError(f"Anthropic refused the credentials ({type(e).__name__}): {e}") from e
        except anthropic.BadRequestError as e:
            # A 400 is deterministic (the same request will fail again); the most common
            # one on a research account is an exhausted credit balance.
            raise FatalApiError(f"Anthropic rejected the request (400): {e}") from e
        except anthropic.NotFoundError as e:
            raise FatalApiError(f"model or endpoint not found: {e}") from e
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise BackendError(f"{type(e).__name__}: {str(e)[:300]}") from e
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return self.parse_message(msg, batch=False, elapsed_ms=elapsed_ms)

    # ------------------------------------------------------------------- selftest

    def selftest(self):
        report = {"model": self.model_cfg.get("name"), "api_model_id": self.model_cfg.get("api_id")}
        try:
            report["api_key"] = "present" if self.api_key else "missing"
        except BackendError as e:
            report["api_key"] = f"FAIL: {e}"
            return report
        try:
            ans, meta = self.generate("Reply with the single digit 7 and nothing else.")
            report["generate"] = f"ok -> {ans!r}"
            report.update({k: meta[k] for k in ("prompt_tokens", "completion_tokens", "finish_reason",
                                                 "cost_usd", "duration_ms", "response_model")})
            report["sampling_sent"] = self.static_metadata()["sampling_sent"]
        except Exception as e:
            report["generate"] = f"FAIL: {type(e).__name__}: {e}"
        return report


_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_shared_anthropic_backend(model_key=None, api_key=None):
    cfg = load_backend_config()
    key = model_key or (cfg.get("anthropic", {}) or {}).get("active_model", "claude-sonnet-5-5")
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = AnthropicBackend(config=cfg, model_key=key, api_key=api_key)
        elif api_key:
            _CACHE[key]._api_key = api_key
        return _CACHE[key]


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Anthropic backend selftest")
    ap.add_argument("--model", default=None, help="model key from backend.yaml anthropic.models")
    ap.add_argument("--api-key", nargs="?", const="__ASK__", default=None,
                    help="API key: a value, '-' for stdin, or the flag alone to be prompted")
    args = ap.parse_args()

    from utils.general_utils import resolve_api_key
    backend = get_shared_anthropic_backend(args.model, resolve_api_key(args.api_key))
    print(json.dumps(backend.selftest(), indent=2, ensure_ascii=False))
