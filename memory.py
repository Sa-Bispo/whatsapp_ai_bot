from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from config import REDIS_URL


_fallback_histories: dict[str, InMemoryChatMessageHistory] = {}


def _get_fallback_history(session_id: str) -> InMemoryChatMessageHistory:
    history = _fallback_histories.get(session_id)
    if history is None:
        history = InMemoryChatMessageHistory()
        _fallback_histories[session_id] = history
    return history


def get_session_history(session_id):
    normalized_session_id = str(session_id or '').strip() or 'default-session'

    if not REDIS_URL:
        return _get_fallback_history(normalized_session_id)

    try:
        history = RedisChatMessageHistory(
            session_id=normalized_session_id,
            url=REDIS_URL,
        )
        # Força uma leitura para validar conectividade e evitar erro tardio em runtime.
        _ = history.messages
        return history
    except Exception:
        return _get_fallback_history(normalized_session_id)
