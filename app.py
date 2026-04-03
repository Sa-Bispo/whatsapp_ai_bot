import re

from fastapi import FastAPI, Request

from message_buffer import buffer_message


app = FastAPI()


def extract_chat_id(payload: dict) -> str | None:
    raw_chat_id = payload.get('data', {}).get('key', {}).get('remoteJid')

    if not raw_chat_id or raw_chat_id.endswith('@g.us'):
        return None

    number = raw_chat_id.split('@')[0]
    number = re.sub(r'\D', '', number)
    return number or None


def extract_message_text(payload: dict) -> str | None:
    message_data = payload.get('data', {}).get('message', {})

    possible_texts = [
        message_data.get('conversation'),
        message_data.get('extendedTextMessage', {}).get('text'),
        message_data.get('imageMessage', {}).get('caption'),
        message_data.get('videoMessage', {}).get('caption'),
    ]

    for text in possible_texts:
        if isinstance(text, str) and text.strip():
            return text.strip()

    return None

@app.post('/webhook')
async def webhook(request: Request):
    data = await request.json()
    event = data.get('event')
    print(f"[WEBHOOK] evento recebido: {event}")
    chat_id = extract_chat_id(data)
    message = extract_message_text(data)

    print(f"[WEBHOOK] chat_id={chat_id} message={message}")

    if chat_id and message:
        await buffer_message(
            chat_id=chat_id,
            message=message,
        )

    return {'status': 'ok'}
