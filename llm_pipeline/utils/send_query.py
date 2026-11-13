import os
import time

from utils.llama_backend import LlamaServerBackend, ContextOverflowError, get_shared_backend
from utils.zai_backend import get_shared_zai_backend, FatalApiError
from utils.anthropic_backend import get_shared_anthropic_backend, RefusalError, MaxTokensError

# The cloud SDKs are imported lazily inside their own _send_* methods: running the
# local llama.cpp backend must not require google-genai/openai/anthropic to be installed.

def get_api_key(provider):
    """Extracts the correct API key from the environment based on the selected provider."""
    from dotenv import load_dotenv
    load_dotenv()
    key_map = {"openai": "OPENAI_API_KEY"}
    if provider not in key_map:
        raise ValueError(f"Unknown provider: {provider}")
    api_key = os.getenv(key_map[provider])
    if not api_key:
        raise RuntimeError(f"API key for {provider} not set")
    return api_key

def get_gemini_keys():
    """ Loads all GEMINI_API_KEY_X entries"""
    from dotenv import load_dotenv
    load_dotenv()

    keys = []
    idx = 0

    while True:
        key = os.getenv(f"GEMINI_API_KEY_{idx}")

        if key is None:
            break

        keys.append(key)
        idx += 1

    if not keys:
        raise RuntimeError("No Gemini API keys found")

    return keys

class LLMSession:
    def __init__(self):
        self._history = []
        self._cache = None
        self._client = None
        self._model = None
        self.MAX_RETRIES = 5
        self.BASE_RETRY_DELAY = 30

        # Gemini key rotation. Loaded on first use so that a local-only run does not
        # abort just because no GEMINI_API_KEY_X is defined.
        self._gemini_keys = None
        self._current_gemini_key_idx = 0

        # llama.cpp backend, built on first use; metadata of the last calls (R7)
        self._llama = None
        self._last_meta = []

    def reset(self):
        """Called between prompts to clear history and delete the cache"""
        if self._cache and self._client:
            try:
                self._client.caches.delete(name=self._cache.name)
            except Exception:
                pass
        self._history = []
        self._cache = None

    def _get_current_gemini_key(self):
        if self._gemini_keys is None:
            self._gemini_keys = get_gemini_keys()
        return self._gemini_keys[self._current_gemini_key_idx]

    def _rotate_gemini_key(self):
        """ Switch to next Gemini key"""

        if self._gemini_keys is None:
            self._gemini_keys = get_gemini_keys()
        old_idx = self._current_gemini_key_idx
        self._current_gemini_key_idx = (self._current_gemini_key_idx + 1) % len(self._gemini_keys)
        print("Switching Gemini key from {old_idx} to {self._current_gemini_key_idx}")

        self._client = None
        self._cache = None

    def _is_rate_limit_error(self, error):
        """ Detect Gemini quota/rate/resource exhaustion errors to determine whether to switch API keys"""
        error_text = str(error).lower()
        rate_limit_indicators = ["429", "resource_exhausted", "quota", "rate limit", "rate_limit", "exceeded", "resource has been exhausted"]
        return any(x in error_text for x in rate_limit_indicators)

    def get_backend(self, model_key=None):
        """
        The llama.cpp backend, shared process-wide. Sessions are per worker thread but
        must not each re-probe and re-hash the GGUF.
        """
        if self._llama is None:
            self._llama = get_shared_backend(model_key)
        return self._llama

    def last_metadata(self):
        """Reproducibility metadata of the most recent send_with_retry (R7)."""
        return list(self._last_meta)

    def send_query(self, provider, model_name, prompt):
        """Handles calling of query functions per provider."""
        if provider == "llamacpp":
            return self._send_llamacpp(model_name, prompt)
        elif provider == "zai":
            return self._send_zai(model_name, prompt)
        elif provider == "gemini":
            return self._send_gemini(model_name, prompt)
        elif provider == "openai":
            api_key = get_api_key(provider)
            return self._send_openai(model_name, prompt, api_key)
        elif provider == "anthropic":
            return self._send_anthropic(model_name, prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def send_with_retry(self, session, provider, model, prompt_parts, run_idx):
        """ Sends all prompt parts with exponential backoff retry on failure"""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                llm_outputs = []
                self._last_meta = []
                for i, part in enumerate(prompt_parts, start=1):
                    output = session.send_query(provider, model, part)
                    llm_outputs.append(output)
                return llm_outputs

            except (ContextOverflowError, FatalApiError, RefusalError, MaxTokensError):
                # None gets better by waiting: a prompt that does not fit will not fit,
                # an exhausted balance will not refill, and a safety decline or a
                # generation that rambles past max_tokens is a property of the prompt. Retrying these burned 450s per
                # case in silence, which is how a whole run could look "busy" while doing
                # nothing at all.
                raise

            except Exception as e:
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY * (2 ** (attempt - 1))  # Keep increasing delay if continuous failure
                    # Say why before sleeping: a silent backoff is indistinguishable from
                    # a run that is simply slow.
                    print(f"Run {run_idx + 1} attempt {attempt}/{self.MAX_RETRIES} failed "
                          f"({type(e).__name__}: {str(e)[:160]}), retrying in {delay}s")
                    session.reset()
                    time.sleep(delay)
                else:
                    print(f"Run {run_idx + 1} failed after {self.MAX_RETRIES} attempts, skipped")
                    return None

    def _send_llamacpp(self, model_name, prompt):
        """
        Send query to a local llama.cpp llama-server.

        model_name selects the entry in config/backend.yaml (primary/secondary). The
        argparse default is Gemini-shaped, so anything unknown falls back to the
        configured active_model rather than failing.
        """
        backend = self.get_backend(model_key=None)
        if model_name in (backend.cfg.get("models") or {}) and model_name != backend.model_key:
            self._llama = get_shared_backend(model_name)
            backend = self._llama

        answer, meta = backend.generate(prompt)
        self._last_meta.append({**backend.static_metadata(), **meta})
        return answer

    def _send_zai(self, model_name, prompt):
        """
        Send query to Z AI (GLM). model_name selects an entry of zai.models in
        config/backend.yaml; anything unrecognised falls back to zai.active_model.
        """
        backend = get_shared_zai_backend(model_name if model_name in ("glm-5.3-flash", "glm-5.2") else None)
        answer, meta = backend.generate(prompt)
        self._last_meta.append({**backend.static_metadata(), **meta})
        return answer

    def _send_gemini(self, model_name, prompt):
        """ Send query to gemini, including possible caching"""
        from google import genai
        from google.genai import types

        full_model = f"gemini-{model_name}"

        api_key = self._get_current_gemini_key()

        if self._client is None:
            self._client = genai.Client(api_key=api_key)
        self._model = full_model

        self._history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        # Send message, either with cache or general history
        try:
            if self._cache:
                response = self._client.models.generate_content(model=full_model, contents=[self._history[-1]], config=types.GenerateContentConfig(cached_content=self._cache.name))
            else:
                response = self._client.models.generate_content(model=full_model, contents=self._history)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

        output = response.text
        self._history.append(types.Content(role="model", parts=[types.Part(text=output)]))

        # Try to create/refresh cache
        try:
            if self._cache:
                self._client.caches.delete(name=self._cache.name)
            self._cache = self._client.caches.create(model=full_model, config=types.CreateCachedContentConfig(contents=self._history, ttl="300s"))
        except Exception:
            # Content too short to cache
            self._cache = None

        return output

    def _send_openai(self, model_name, prompt, api_key):
        """ Send query to chatgpt. TODO no caching yet"""
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    def _send_anthropic(self, model_name, prompt):
        """
        Send query to Anthropic (Claude). model_name selects an entry of
        anthropic.models in config/backend.yaml; anything unrecognised falls back to
        anthropic.active_model. Key, pricing and R7 metadata live in the backend.
        """
        from utils.anthropic_backend import load_backend_config
        known = (load_backend_config().get("anthropic", {}) or {}).get("models", {})
        backend = get_shared_anthropic_backend(model_name if model_name in known else None)
        answer, meta = backend.generate(prompt)
        self._last_meta.append({**backend.static_metadata(), **meta})
        return answer