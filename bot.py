import logging
from functools import lru_cache
from typing import List, Dict

import asyncio
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

from openai import AsyncOpenAI
# tiktoken is used only for token counting
import tiktoken  

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

LM_BASE        = "http://LMSTUDIOIP:1234/v1"       # LM Studio server
MODEL_NAME     = "ai"
TELEGRAM_TOKEN = "TGTOKEN"
SYSTEM_PROMPT = (
    "Respond concisely (<100 words) unless asked for detail."
)

MAX_TOKENS_CONTEXT = 4096       # how many *prompt* tokens we keep
MAX_TOKENS_REPLY   = 300        # how long model responses can be

TEMPERATURE = 0.7               # tweak if needed

# ────────────────────────────────────────────────────────────
# LOGGING
# ────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# TOKEN‑COUNT HELPERS
# ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_encoder():
    """
    Returns a tiktoken encoder suitable for chat model.
    Falls back to cl100k_base (same as GPT3.5/4) if the model
    name isn't recognised - works well with ChatML.
    """
    try:
        return tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def num_tokens(msgs: List[Dict[str, str]]) -> int:
    #"""
    # *Approximate* token count for OpenAI ChatML format.
    #Adjustment: 4 tokens per message (role & formatting) + 2 priming tokens.
    # """
    enc = _get_encoder()
    tokens = 0
    for m in msgs:
        tokens += 4                          
        tokens += len(enc.encode(m["content"]))
    tokens += 2                              
    return tokens

def trim_to_limit(msgs: List[Dict[str, str]],
                  limit: int = MAX_TOKENS_CONTEXT) -> None:
    #
    # Mutates `msgs` in‑place, deleting the oldest non‑system messages
    # until the token count is ≤ limit.
    # Keeps *exactly* the first element (assumed to be the system prompt).
    #
    while len(msgs) > 2 and num_tokens(msgs) > limit:
        del msgs[1]          # drop the oldest after system prompt

# ────────────────────────────────────────────────────────────
# MAIN HANDLER
# ────────────────────────────────────────────────────────────

# Async client for the local LM Studio / OpenAI‑compatible server
client = AsyncOpenAI(base_url=LM_BASE, api_key="local")  # api_key value is ignored by LM Studio

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle every incoming text message (group or private)."""
    msg = update.effective_message
    if not msg or not msg.text:
        return

    # 1) Show “typing…”
    await context.bot.send_chat_action(chat_id=msg.chat_id,
                                       action=ChatAction.TYPING)

    # 2) Fetch or initialise conversation history for this chat
    history: List[Dict[str, str]] = context.chat_data.get("history", [])
    if not history:
        # first round in this chat – insert system prompt
        history.append({"role": "system", "content": SYSTEM_PROMPT})

    # 3) Append the user's message
    history.append({"role": "user", "content": msg.text})

    # 4) Trim history to ≤ 4 K tokens
    trim_to_limit(history)

    try:
        # 5) Call the model (single blocking request – no streaming)
        completion = await client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = history,
            max_tokens  = MAX_TOKENS_REPLY,
            temperature = TEMPERATURE,
        )
        reply = completion.choices[0].message.content.strip()

    except Exception as err:
        logger.exception(err)
        await msg.reply_text("ooops") # cannot reach the local model
        return

    # 6) Send the reply
    await msg.reply_text(reply)

    # 7) Record assistant message and store back
    history.append({"role": "assistant", "content": reply})
    context.chat_data["history"] = history

# ────────────────────────────────────────────────────────────
# BOT ENTRY‑POINT
# ────────────────────────────────────────────────────────────

def main() -> None:
    """Run the bot until Ctrl-C."""
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Handle every non‑command text message
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                   handle_msg))

    # Start polling (blocking; Application handles its own event‑loop)
    logger.info("Bot started – polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
