"""
LLM caller via OpenRouter API.
Model: meta-llama/llama-3-8b-instruct
"""

import os
import requests
from typing import List, Dict

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3-8b-instruct"


def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Send messages to OpenRouter and return assistant reply.
    Raises ValueError on missing key or API errors.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Please add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "DocuMind AI",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        raise ValueError("Request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 401:
            raise ValueError("Invalid API key. Check your OPENROUTER_API_KEY.")
        elif status == 429:
            raise ValueError("Rate limit exceeded. Please wait and retry.")
        else:
            raise ValueError(f"API error {status}: {e}")
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")
