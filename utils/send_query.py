from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import time

load_dotenv()
MAX_RETRIES = 5
BASE_RETRY_DELAY = 30
KEY_MAP = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    

def get_api_key(provider):
    """Extracts the correct API key from the environment based on the selected provider."""    
    if provider not in KEY_MAP:
        raise ValueError(f"Unknown provider: {provider}")
    api_key = os.getenv(KEY_MAP[provider])
    if not api_key:
        raise RuntimeError(f"API key for {provider} not set")
    return api_key

def send_query(provider, model_name, prompt):
        """Handles calling of query functions per provider."""
        api_key = get_api_key(provider)
        if provider == "gemini":
            return send_gemini(model_name, prompt, api_key)
        elif provider == "openai":
            return send_openai(model_name, prompt, api_key)
        elif provider == "anthropic":
            return send_anthropic(model_name, prompt, api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
def send_with_retry(provider, model, prompt_text, run_idx):
    """ Sends prompt with exponential backoff retry on failure"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return send_query(provider, model, prompt_text)
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))  # Keep increasing delay if continuous failure
                time.sleep(delay)
            else:
                print(f"Run {run_idx + 1} failed after {MAX_RETRIES} attempts, skipped")
                return None

def send_gemini(model_name, prompt, api_key):
    """ Send query to gemini"""
    full_model = f"gemini-{model_name}"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=full_model, contents=[types.Content(role="user", parts=[types.Part(text=prompt)])])
    return response.text

def send_openai(model_name, prompt, api_key):
    """ Send query to chatgpt"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def send_anthropic(model_name, prompt, api_key):
    """ Send query to claude"""
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(model=model_name, max_tokens = 4096, messages=[{"role": "user", "content": prompt}])
    return "".join(block.text for block in response.content if block.type == "text")