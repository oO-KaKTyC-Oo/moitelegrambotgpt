import asyncio
import contextlib
import json
import logging
import os
import re
import time

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import BaseFilter, Command
from aiogram.types import Message

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # .env не обязателен: переменные можно задать в окружении
    pass

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

_missing = [
    name
    for name, value in (
        ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
        ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
    )
    if not value
]
if _missing:
    raise SystemExit(
        "Не заданы переменные окружения: " + ", ".join(_missing)
        + ". Скопируй .env.example в .env и заполни, либо задай их в панели хостинга."
    )

_raw_owner = os.getenv("TELEGRAM_OWNER_ID", "").strip()
if _raw_owner and not _raw_owner.isdigit():
    raise SystemExit("TELEGRAM_OWNER_ID должен быть числом (твой Telegram ID), например 123456789")
OWNER_ID: int | None = int(_raw_owner) if _raw_owner else None
if OWNER_ID is None:
    logging.warning(
        "TELEGRAM_OWNER_ID не задан: бот отвечает всем. Отправь боту /id и впиши ID в .env"
    )

MODEL = os.getenv("MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_HISTORY = 20  # сколько последних сообщений помнить

SYSTEM_PROMPT = """Ты - Honest Challenger: честный, критически мыслящий собеседник. Истина и точность важнее вежливого согласия.

Правила по сути:
1. Проверяй предпосылки пользователя. Если посылка неверна, скажи об этом в первом предложении.
2. Не поддакивай и не льсти. Сразу к сути.
3. Спорь аргументированно: указывай на слабые места и приводи контраргументы.
4. Отделяй факты от мнений. Если не уверен в факте, так и скажи, ничего не выдумывай.
5. Стой на своём, если прав. Признавай правоту пользователя только когда его аргументы верны.
Тон спокойный и конструктивный, без грубости.

Правила по форме (строго):
- Отвечай коротко: обычно 2-5 предложений. Длинно пиши только если пользователь прямо просит подробности.
- Пиши простым текстом, как в обычной переписке. Никакого markdown: не используй звёздочки, решётки, обратные кавычки, линии из дефисов, заголовки, таблицы и жирный шрифт.
- Списки только если они действительно нужны, тогда каждый пункт с новой строки и начинается с "- ".
- Не повторяй вопрос пользователя и не добавляй вступлений и заключений.
- Отвечай на языке пользователя."""

bot = Bot(TELEGRAM_TOKEN)
dp = Dispatcher()

histories: dict[int, list[dict]] = {}
locks: dict[int, asyncio.Lock] = {}


class OwnerFilter(BaseFilter):
    """Пускает только владельца, если задан TELEGRAM_OWNER_ID."""

    async def __call__(self, message: Message) -> bool:
        if OWNER_ID is None:
            return True
        if message.from_user and message.from_user.id == OWNER_ID:
            return True
        logging.info(
            "Игнорирую сообщение от id=%s username=@%s: бот привязан к владельцу",
            message.from_user.id if message.from_user else "?",
            message.from_user.username if message.from_user else "?",
        )
        return False


def trim_history(history: list[dict]) -> None:
    """Оставляет последние MAX_HISTORY сообщений, начиная с реплики пользователя."""
    del history[:-MAX_HISTORY]
    while history and history[0].get("role") != "user":
        history.pop(0)
    # reasoning_details нужны только у последнего ответа модели
    for msg in history[:-1]:
        msg.pop("reasoning_details", None)


async def ask_ai(session: aiohttp.ClientSession, history: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "reasoning": {"enabled": True},
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    last_error = "неизвестная ошибка"
    for attempt in range(3):
        try:
            async with session.post(API_URL, json=payload, headers=headers) as resp:
                body = await resp.text()
                if resp.status == 200:
                    return json.loads(body)["choices"][0]["message"]
                last_error = f"HTTP {resp.status}: {body[:300]}"
                if resp.status != 429 and resp.status < 500:
                    raise RuntimeError(f"OpenRouter {last_error}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 2:
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenRouter недоступен: {last_error}")


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _chars_for(text: str, limit: int) -> int:
    """Сколько символов с начала text влезают в limit единиц UTF-16."""
    units = 0
    for i, ch in enumerate(text):
        units += 2 if ord(ch) > 0xFFFF else 1
        if units > limit:
            return i
    return len(text)


def split_text(text: str, limit: int = 3800):
    """Режет текст на куски, влезающие в лимит Telegram (UTF-16 units)."""
    while text:
        cut = _chars_for(text, limit)
        if cut >= len(text):
            if text.strip():
                yield text
            return
        newline = text.rfind("\n", 0, cut)
        if newline > 0 and newline >= cut // 2:
            cut = newline
        chunk = text[:cut].strip("\n")
        text = text[cut:].lstrip("\n")
        if chunk.strip():
            yield chunk


def clean_markdown(text: str) -> str:
    text = re.sub(r"^[ \t]*([-*_][ \t]*){3,}$", "", text, flags=re.M)  # линии ---, ***
    text = re.sub(r"^[ \t]*\*[ \t]+", "- ", text, flags=re.M)          # "* пункт" -> "- пункт"
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text, flags=re.S)        # **жирный**
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", text)    # *курсив* внутри строки
    text = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"\1", text)      # _курсив_
    text = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text)  # [текст](url)
    text = text.replace("**", "")                                       # одиночные хвосты
    text = re.sub(r"^#{1,6}[ \t]*", "", text, flags=re.M)               # заголовки #
    text = text.replace("`", "")                                        # обратные кавычки
    text = re.sub(r"\n{3,}", "\n\n", text)                              # лишние пустые строки
    return text.strip()


async def keep_typing(chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        raise
    except TelegramAPIError:
        logging.debug("Не удалось отправить typing в чат %s", chat_id)


@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "Я ИИ-собеседник, который не поддакивает и проверяет факты. "
        "Пиши вопрос или идею. /reset очищает историю диалога."
    )


@dp.message(Command("id"))
async def my_id(message: Message):
    await message.answer(f"Твой Telegram ID: {message.from_user.id}")


@dp.message(Command("reset"), OwnerFilter())
async def reset(message: Message):
    histories.pop(message.chat.id, None)
    await message.answer("История очищена.")


@dp.message(F.text, OwnerFilter())
async def handle(message: Message, session: aiohttp.ClientSession):
    chat_id = message.chat.id
    lock = locks.setdefault(chat_id, asyncio.Lock())

    async with lock:  # ответы этого чата не перемешиваются
        history = histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": message.text})
        started = time.monotonic()
        logging.info("Запрос от %s в чате %s", message.from_user.id, chat_id)

        typing = asyncio.create_task(keep_typing(chat_id))
        try:
            reply = await ask_ai(session, history[-MAX_HISTORY:])
        except Exception:
            logging.exception("AI request failed")
            history.pop()  # убираем вопрос, на который не получили ответ
            await message.answer("Не удалось получить ответ от ИИ. Попробуй ещё раз позже.")
            return
        finally:
            typing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing

        content = clean_markdown(reply.get("content") or "")
        if not content:
            history.pop()
            await message.answer("Модель вернула пустой ответ. Попробуй переформулировать.")
            return

        assistant_msg = {"role": "assistant", "content": content}
        if reply.get("reasoning_details"):
            assistant_msg["reasoning_details"] = reply["reasoning_details"]  # без изменений
        history.append(assistant_msg)
        trim_history(history)

        for chunk in split_text(content):
            try:
                await message.answer(chunk)
            except TelegramAPIError:
                logging.exception("Не удалось отправить часть ответа")
        logging.info("Ответ отправлен за %.1f с", time.monotonic() - started)


@dp.message(OwnerFilter())
async def non_text(message: Message):
    await message.answer("Пока понимаю только текст: пришли вопрос сообщением.")


async def main():
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await dp.start_polling(bot, session=session)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
