"""Telegram Bot API helper utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from .config import settings

logger = logging.getLogger(__name__)
TELEGRAM_MAX_MESSAGE_LENGTH = 4096


@dataclass(slots=True)
class TelegramSendResult:
    sent: bool
    error_message: str | None = None


def truncate_telegram_message(text: str, *, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> str:
    """Trim text to Telegram's message length without splitting awkwardly."""

    text = text.strip()
    if len(text) <= limit:
        return text
    suffix = "\n…"
    return text[: limit - len(suffix)].rstrip() + suffix


class TelegramClient:
    """Small Telegram Bot API client that fails gracefully when unconfigured."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None, *, timeout: float = 10.0) -> None:
        self.bot_token = bot_token if bot_token is not None else settings.telegram_bot_token
        self.chat_id = chat_id if chat_id is not None else settings.telegram_chat_id
        self.timeout = timeout

    def send_message(self, text: str, *, chat_id: str | None = None) -> TelegramSendResult:
        text = truncate_telegram_message(text)
        target_chat_id = chat_id or self.chat_id
        if not text:
            return TelegramSendResult(False, "Telegram message is empty")
        if not self.bot_token or not target_chat_id:
            logger.info("Telegram send skipped because TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing")
            return TelegramSendResult(False, "Telegram bot token or chat id is not configured")
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": target_chat_id, "text": text, "disable_web_page_preview": True},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Telegram send failed")
            return TelegramSendResult(False, str(exc))
        return TelegramSendResult(True)
