import asyncio
import logging
import random
import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, List

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

from openai import AsyncOpenAI
import tiktoken  # only for token counting

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────
LM_BASE        = "http://LM-STUDIOHOST:1234/v1"       # LM Studio server
MODEL_NAME     = "ai"
TELEGRAM_TOKEN = "TOKEN"
SYSTEM_PROMPT = (
    "You are ...."
    "Your name is ....."
)

MAX_TOKENS_CONTEXT = 4096          # tokens we keep in the prompt
MAX_TOKENS_REPLY = 300             # model reply length
TEMPERATURE = 0.7

RESET_TRIGGER = "/done"           # phrase that wipes context
PRESET_RESPONSES = [               # bot replies when resetting
    "Ok, I have reset the context.",
    "Okay, I will reset the context now.",
    "Okay, I have cleared the context."
]

LOG_FILE = "chat_log.txt"          # where all turns are appended

# ────────────────────────────────────────────────────────────
# LOGGING (stdout for diagnostics, separate file for content)
# ────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _log_turn(role: str, chat_id: int, text: str) -> None:
    line = (
        f"{datetime.utcnow().isoformat(timespec='seconds')}Z | "
        f"{chat_id} | {role:<9} | {text.replace('\n', ' ')}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line)


# ────────────────────────────────────────────────────────────
# TOKEN‑COUNT HELPERS
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_encoder():
    try:
        return tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def num_tokens(msgs: List[Dict[str, str]]) -> int:
    enc = _get_encoder()
    tokens = 0
    for m in msgs:
        tokens += 4
        tokens += len(enc.encode(m["content"]))
    tokens += 2
    return tokens


def trim_to_limit(msgs: List[Dict[str, str]], limit: int = MAX_TOKENS_CONTEXT) -> None:
    # """Mutates `msgs`, deleting oldest non - system messages until ≤ limit."""
    while len(msgs) > 2 and num_tokens(msgs) > limit:
        del msgs[1]  # drop the oldest after system prompt


# ────────────────────────────────────────────────────────────
# TEXT‑SPLITTING: send 1–2 messages with pauses
# ────────────────────────────────────────────────────────────
def _split_reply(text: str, force_two: bool = False) -> List[str]:
    #"""
    #Return one or two chunks, always ending on sentence boundaries.
    #• If the reply is short (≤250 chars) and force_two is False → one chunk.
    #• Otherwise we split after the last full sentence that keeps
    #  the first chunk ≈½ of the original length.
    #"""
    if not force_two and len(text) <= 250:
        return [text]

    # Split into sentences using simple punctuation look‑behind
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) == 1:                     # no punctuation found
        cut = text.rfind(" ", 0, len(text) // 2)  # fallback: nearest space
        return [text[:cut].strip(), text[cut + 1 :].strip()]

    half_len  = len(text) / 2
    part1, part2 = [], []
    char_count = 0

    for s in sentences:
        # Keep adding sentences to part1 until we cross halfway
        if char_count < half_len or not part1:
            part1.append(s)
            char_count += len(s) + 1            # +1 for a space/newline
        else:
            part2.append(s)

    # Edge case: everything landed in part1 → move last sentence to part2
    if not part2:
        part2.append(part1.pop())

    return [" ".join(part1).strip(), " ".join(part2).strip()]


# ────────────────────────────────────────────────────────────
# MAIN HANDLER
# ────────────────────────────────────────────────────────────
client = AsyncOpenAI(base_url=LM_BASE, api_key="local")  # api_key is ignored


async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.text:
        return

    chat_id = msg.chat_id
    user_text = msg.text.strip()
    _log_turn("user", chat_id, user_text)

    # ── 0) RESET TRIGGER ────────────────────────────────────
    if RESET_TRIGGER.lower() in user_text.lower():
        context.chat_data["history"] = []  # wipe
        reply = random.choice(PRESET_RESPONSES)
        await msg.reply_text(reply)
        _log_turn("assistant", chat_id, reply)
        return

    # ── 1) Typing indicator ─────────────────────────────────
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # ── 2) History init/fetch ───────────────────────────────
    history: List[Dict[str, str]] = context.chat_data.get("history", [])
    if not history:
        history.append({"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_text})
    trim_to_limit(history)

    # ── 3) Call the LLM ─────────────────────────────────────
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=history,
            max_tokens=MAX_TOKENS_REPLY,
            temperature=TEMPERATURE,
        )
        full_reply = completion.choices[0].message.content.strip()
    except Exception as err:
        logger.exception(err)
        await msg.reply_text("Oops")
        return

    # ── 4) Send reply in 1–2 chunks with delays ─────────────
    chunks = _split_reply(full_reply)
    for ix, chunk in enumerate(chunks):
        if ix:  # pause before second message
            await asyncio.sleep(random.uniform(0.4, 1.1))
            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )
        await msg.reply_text(chunk)
        _log_turn("assistant", chat_id, chunk)

    # ── 5) Save combined reply to history ───────────────────
    history.append({"role": "assistant", "content": full_reply})
    context.chat_data["history"] = history


# ────────────────────────────────────────────────────────────
# BOT ENTRY‑POINT
# ────────────────────────────────────────────────────────────
def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))

    logger.info("Bot started - polling…")
    app.run_polling()


if __name__ == "__main__":
    main()
