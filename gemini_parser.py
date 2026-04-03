import asyncio
import json
import os
import re

from google import genai
from google.genai import types


_GEMINI_MODEL_NAME = 'gemini-2.5-flash'
_GEMINI_SYSTEM_INSTRUCTION = (
    'You are an expert e-commerce NLU assistant. '
    'CRITICAL RULES: '
    '1. Extract ALL products requested by the user. Do not leave any requested item behind. '
    '2. You MUST strictly map the user\'s request to the exact "codigo_pai" provided in the CATALOGO_ATUAL_JSON. If the user uses slang or typos, find the closest matching product in the catalog. '
    '3. If the user does not specify a variation, but the catalog shows only "Unico", use "Unico". '
    '4. Return ONLY a valid JSON array of objects. No markdown, no prose. '
    'Format: {"codigo_pai": "string", "variacao": "string", "quantidade": number}. '
    'If absolutely nothing matches, return [].'
)


def _extract_json_array(text: str) -> list:
    raw = (text or '').strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw, flags=re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


_NLU_SYSTEM_INSTRUCTION = (
    'Você é o analisador NLU de uma loja de suplementos. '
    'Sua função é ler a mensagem do usuário, analisar o carrinho atual e retornar APENAS um JSON estrito. '
    'Regras: '
    '1. Identifique a intenção: "duvida_tecnica", "adicionar_carrinho", "ver_carrinho", "checkout", "atendimento_humano". '
    '2. Se a intenção for "adicionar_carrinho", identifique categoria, marca, sabor e tamanho do produto. '
    '3. Tente mapear o produto para um "codigo_pai" e "variacao" exatos do CATALOGO_ATUAL_JSON fornecido. '
    '4. Se o produto estiver completamente identificado (codigo_pai mapeado), defina "status_item": "completo". '
    '5. Se faltarem dados essenciais (marca, sabor ou tamanho quando necessário), '
    'liste os campos faltantes em "dados_faltantes" e defina "status_item": "incompleto". '
    '6. Se "status_item" for "completo", sugira um produto complementar em "upsell_sugerido" (ex: Whey -> Creatina). '
    '7. Para outras intenções, defina "status_item": null e "produto_identificado": null. '
    '8. Retorne APENAS JSON válido, sem markdown e sem texto adicional. '
    'Formato de saída: '
    '{"intencao": "string", '
    '"produto_identificado": {"categoria": "string|null", "marca": "string|null", '
    '"sabor": "string|null", "tamanho": "string|null", '
    '"codigo_pai": "string|null", "variacao": "string|null", "quantidade": 1}, '
    '"dados_faltantes": [], "status_item": "completo|incompleto|null", "upsell_sugerido": "string|null"}'
)


def _extract_json_object(text: str) -> dict:
    raw = (text or '').strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


async def analyze_message(user_message: str, cart_items: list[dict], catalog: list[dict]) -> dict:
    api_key = os.getenv('GEMINI_API_KEY', '').strip()
    if not api_key:
        print('[GEMINI_PARSER] GEMINI_API_KEY not set. Returning fallback intent.')
        return {'intencao': 'desconhecido'}

    client = genai.Client(api_key=api_key)

    prompt = (
        'CARRINHO_ATUAL_JSON:\n'
        f'{json.dumps(cart_items, ensure_ascii=False)}\n\n'
        'CATALOGO_ATUAL_JSON:\n'
        f'{json.dumps(catalog, ensure_ascii=False)}\n\n'
        'MENSAGEM_USUARIO:\n'
        f'{user_message}\n\n'
        'Retorne APENAS o JSON de análise NLU conforme o formato especificado.'
    )

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=_GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_NLU_SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=600,
            ),
        )
    except Exception as e:
        print(f'[GEMINI_PARSER] Exception in analyze_message: {e!r}')
        return {'intencao': 'desconhecido'}

    response_text = getattr(response, 'text', '') or ''
    print(f'[GEMINI_PARSER] NLU response: {response_text!r}')
    return _extract_json_object(response_text)


def _normalize_items(items: list) -> list[dict]:
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        codigo_pai = str(item.get('codigo_pai', '')).strip()
        variacao = str(item.get('variacao', '')).strip()

        quantidade_raw = item.get('quantidade', 1)
        try:
            quantidade = int(quantidade_raw)
        except (TypeError, ValueError):
            continue

        if not codigo_pai or quantidade <= 0:
            continue

        normalized.append(
            {
                'codigo_pai': codigo_pai,
                'variacao': variacao,
                'quantidade': quantidade,
            }
        )

    return normalized


async def extract_order_intent(user_message: str, catalogo: list) -> list[dict]:
    api_key = os.getenv('GEMINI_API_KEY', '').strip()
    if not api_key:
        print('[GEMINI_PARSER] GEMINI_API_KEY not set. Returning empty list.')
        return []

    client = genai.Client(api_key=api_key)

    prompt = (
        'CATALOGO_ATUAL_JSON:\n'
        f'{json.dumps(catalogo, ensure_ascii=False)}\n\n'
        'MENSAGEM_USUARIO:\n'
        f'{user_message}\n\n'
        'INSTRUCAO:\n'
        'Extraia os itens solicitados e responda APENAS com JSON válido (lista).'
    )

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=_GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_GEMINI_SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=800,
            ),
        )
    except Exception as e:
        print(f'[GEMINI_PARSER] Exception while calling Gemini: {e!r}')
        return []

    response_text = getattr(response, 'text', '') or ''
    print(f'[GEMINI_PARSER] Raw response text: {response_text!r}')
    items = _extract_json_array(response_text)
    return _normalize_items(items)
