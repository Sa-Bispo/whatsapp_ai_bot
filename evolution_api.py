import base64
import mimetypes
from pathlib import Path

import requests

from config import (
    EVOLUTION_API_URL,
    EVOLUTION_INSTANCE_NAME,
    EVOLUTION_AUTHENTICATION_API_KEY,
)


def _resolve_instance_name(instance_name: str | None) -> str:
    instance = (instance_name or EVOLUTION_INSTANCE_NAME or '').strip()
    if not instance:
        raise RuntimeError('EVOLUTION_INSTANCE_NAME não configurado e instance_name não informado.')
    return instance


def send_whatsapp_message(number, text, instance_name=None):
    instance = _resolve_instance_name(instance_name)
    url = f'{EVOLUTION_API_URL}/message/sendText/{instance}'
    headers = {
        'apikey': EVOLUTION_AUTHENTICATION_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {
        'number': number,
        'text': text,
    }
    response = requests.post(
        url=url,
        json=payload,
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()


def send_whatsapp_presence(number, presence='composing', delay=2500, instance_name=None):
    instance = _resolve_instance_name(instance_name)
    url = f'{EVOLUTION_API_URL}/chat/sendPresence/{instance}'
    headers = {
        'apikey': EVOLUTION_AUTHENTICATION_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {
        'number': number,
        'presence': presence,
        'delay': delay,
    }
    response = requests.post(
        url=url,
        json=payload,
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()


def send_whatsapp_media(number, caption, media, mediatype='image', file_name=None, mimetype=None, instance_name=None):
    instance = _resolve_instance_name(instance_name)
    url = f'{EVOLUTION_API_URL}/message/sendMedia/{instance}'
    headers = {
        'apikey': EVOLUTION_AUTHENTICATION_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {
        'number': number,
        'mediatype': mediatype,
        'media': media,
    }

    if caption:
        payload['caption'] = caption

    if file_name:
        payload['fileName'] = file_name

    if mimetype:
        payload['mimetype'] = mimetype

    response = requests.post(
        url=url,
        json=payload,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()


def send_whatsapp_image_file(number, file_path, caption='', instance_name=None):
    path = Path(file_path)

    with path.open('rb') as image_file:
        encoded_media = base64.b64encode(image_file.read()).decode('utf-8')

    detected_mimetype = mimetypes.guess_type(path.name)[0] or 'image/png'

    send_whatsapp_media(
        number=number,
        caption=caption,
        media=encoded_media,
        mediatype='image',
        file_name=path.name,
        mimetype=detected_mimetype,
        instance_name=instance_name,
    )
