import base64
import mimetypes
from pathlib import Path

import requests

from config import (
    EVOLUTION_API_URL,
    EVOLUTION_INSTANCE_NAME,
    EVOLUTION_AUTHENTICATION_API_KEY,
)


def send_whatsapp_message(number, text):
    url = f'{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE_NAME}'
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


def send_whatsapp_presence(number, presence='composing', delay=2500):
    url = f'{EVOLUTION_API_URL}/chat/sendPresence/{EVOLUTION_INSTANCE_NAME}'
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


def send_whatsapp_media(number, caption, media, mediatype='image', file_name=None, mimetype=None):
    url = f'{EVOLUTION_API_URL}/message/sendMedia/{EVOLUTION_INSTANCE_NAME}'
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


def send_whatsapp_image_file(number, file_path, caption=''):
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
    )
