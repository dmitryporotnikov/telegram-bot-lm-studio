# Telegram bot for LM Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](bot.py)
[![LM Studio](https://img.shields.io/badge/LM%20Studio-OpenAI%20API-412991)](https://lmstudio.ai/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?logo=telegram&logoColor=white)](https://core.telegram.org/bots)

A small Telegram bot that talks to a model running in LM Studio. It keeps per-chat context, trims the oldest turns when the prompt approaches the token limit, and appends every turn to `chat_log.txt` so you can read back what was said.

## Setup

Everything is configured in the constants at the top of `bot.py`:

| Constant | What it means |
|---|---|
| `LM_BASE` | LM Studio server, e.g. `http://192.168.1.50:1234/v1`. |
| `MODEL_NAME` | Model identifier as LM Studio reports it. `ai` is a placeholder. |
| `TELEGRAM_TOKEN` | Token from [@BotFather](https://t.me/BotFather). |
| `SYSTEM_PROMPT` | The persona and instructions the model starts with. |
| `MAX_TOKENS_CONTEXT` | Prompt budget, 4096 by default. |
| `MAX_TOKENS_REPLY` | Reply length cap, 300 by default. |
| `TEMPERATURE` | 0.7 by default. |
| `RESET_TRIGGER` | `/done` wipes the conversation. |

Install and run:

```bash
pip install -r requirements.txt
python bot.py
```

LM Studio has to be listening on the network with its OpenAI-compatible server enabled, and the model name in `MODEL_NAME` has to match the one loaded there.

Token counting uses `tiktoken`, falling back to `cl100k_base` when the model name isn't recognized, so the context budget is an estimate rather than an exact count for every model.

There's no license file in this repo yet, so treat the code as all rights reserved.
