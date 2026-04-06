import asyncio
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from chains import generate_persona_response
from database_api import get_tenant_by_instance, get_tenant_configs
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


def extract_instance_name(payload: dict) -> str | None:
    instance = payload.get('instance')

    if isinstance(instance, str) and instance.strip():
        return instance.strip()

    if isinstance(instance, dict):
        nested_name = instance.get('instanceName') or instance.get('name')
        if isinstance(nested_name, str) and nested_name.strip():
            return nested_name.strip()

    data_instance = payload.get('data', {}).get('instance')
    if isinstance(data_instance, str) and data_instance.strip():
        return data_instance.strip()

    return None

@app.post('/api/chat-sync')
async def chat_sync(request: Request):
    """Porta síncrona para o simulador da Landing Page.
    Recebe tenant_id + message, carrega a persona do banco e devolve a
    resposta da IA sem disparar nada na Evolution API.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({'error': 'JSON inválido.'}, status_code=400)

    tenant_id = (data.get('tenant_id') or '').strip()
    message = (data.get('message') or '').strip()
    # session_id permite manter histórico separado por visitante.
    session_id = (data.get('session_id') or tenant_id).strip()

    if not tenant_id or not message:
        return JSONResponse(
            {'error': 'Os campos tenant_id e message são obrigatórios.'},
            status_code=422,
        )

    try:
        configs = await get_tenant_configs(tenant_id)
        prompt_ia = configs.get('promptIa') or ''
        bot_objective = configs.get('botObjective') or 'FECHAR_PEDIDO'

        # generate_persona_response é síncrono (Gemini/OpenAI blocking I/O).
        # Executar em thread pool para não travar o event loop do FastAPI.
        reply: str = await asyncio.to_thread(
            generate_persona_response,
            'Responda à mensagem do usuário de forma natural, seguindo sua persona.',
            message,
            session_id,
            prompt_ia or None,
            bot_objective,
        )
    except Exception as exc:
        print(f'[chat-sync] erro ao gerar resposta: {exc}')
        return JSONResponse(
            {'error': 'Erro interno ao processar a mensagem.'},
            status_code=500,
        )

    return {'reply': reply}


@app.post('/webhook')
async def webhook(request: Request):
    data = await request.json()
    event = data.get('event')
    instance_name = extract_instance_name(data)

    if not instance_name:
        return {'status': 'ignored'}

    tenant_id = await get_tenant_by_instance(instance_name)
    if not tenant_id:
        # Ignora mensagens de instâncias não cadastradas no SaaS.
        return {'status': 'ignored'}

    print(f"[WEBHOOK] evento recebido: {event}")
    chat_id = extract_chat_id(data)
    message = extract_message_text(data)

    print(f"[WEBHOOK] instance={instance_name} tenant_id={tenant_id} chat_id={chat_id} message={message}")

    if chat_id and message:
        await buffer_message(
            chat_id=chat_id,
            message=message,
            tenant_id=tenant_id,
            instance_name=instance_name,
        )

    return {'status': 'ok'}
