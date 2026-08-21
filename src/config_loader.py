from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

with open(BASE_DIR / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)

with open(BASE_DIR / "config" / "prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# exhausted tokens so couldnt continue with open ai models
# def get_llm() -> ChatOpenAI:
#     model = os.getenv("OPENAI_MODEL")
#     if not model:
#         raise RuntimeError(
#             "OPENAI_MODEL is not set. Copy .env.template to .env and set "
#             "a model available to your account."
#         )
#     if not os.getenv("OPENAI_API_KEY"):
#         raise RuntimeError(
#             "OPENAI_API_KEY is not set. Add it to the project-root .env file."
#         )
#
#     return ChatOpenAI(
#         model=model,
#         temperature=float(SETTINGS["model"].get("temperature", 0)),
#     )

def get_llm() -> ChatGoogleGenerativeAI:

    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash-lite",
    )

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Add it to the project-root .env file."
        )

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=float(
            SETTINGS["model"].get(
                "temperature",
                0,
            )
        ),
    )
