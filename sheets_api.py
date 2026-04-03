from __future__ import annotations

from datetime import datetime, timezone
import os
import re

import gspread

from config import (
    GOOGLE_SHEETS_CLIENTES_TAB,
    GOOGLE_SHEETS_CREDENTIALS_PATH,
    GOOGLE_SHEETS_ESTOQUE_TAB,
    GOOGLE_SHEETS_PEDIDOS_TAB,
    GOOGLE_SHEETS_SPREADSHEET_ID,
    GOOGLE_SHEETS_TOKEN_PATH,
)


_gspread_client: gspread.Client | None = None


def _normalize_phone(value: str) -> str:
    return re.sub(r'\D', '', value or '')


def _parse_brl_price(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value or '').strip()
    text = text.replace('R$', '').replace(' ', '')
    text = re.sub(r'[^\d,.-]', '', text)

    if not text:
        raise ValueError('Preço vazio')

    if ',' in text and '.' in text:
        text = text.replace('.', '').replace(',', '.')
    elif ',' in text:
        text = text.replace(',', '.')
    elif text.count('.') > 1:
        integer_part, decimal_part = text.rsplit('.', 1)
        text = integer_part.replace('.', '') + '.' + decimal_part

    return float(text)


def _get_client() -> gspread.Client:
    global _gspread_client

    if _gspread_client:
        return _gspread_client

    credentials_filename = GOOGLE_SHEETS_CREDENTIALS_PATH or 'credentials.json'
    authorized_user_filename = GOOGLE_SHEETS_TOKEN_PATH or 'token.json'

    if not os.path.exists(credentials_filename):
        raise FileNotFoundError(
            f'Arquivo de credenciais OAuth não encontrado: {credentials_filename}'
        )

    token_exists = os.path.exists(authorized_user_filename)

    try:
        _gspread_client = gspread.oauth(
            credentials_filename=credentials_filename,
            authorized_user_filename=authorized_user_filename,
        )
        return _gspread_client
    except Exception as error:
        if not token_exists:
            raise RuntimeError(
                'Token OAuth não encontrado e o fluxo de autorização falhou. '
                'Execute localmente com navegador para gerar o token.json inicial.'
            ) from error
        raise


def _get_worksheet(tab_name: str):
    client = _get_client()
    spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
    return spreadsheet.worksheet(tab_name)


def get_cliente_by_phone(phone: str) -> dict | None:
    normalized_phone = _normalize_phone(phone)
    if not normalized_phone:
        return None

    worksheet = _get_worksheet(GOOGLE_SHEETS_CLIENTES_TAB)
    records = worksheet.get_all_records()

    for record in records:
        record_phone = _normalize_phone(str(record.get('numero', '')))
        if record_phone == normalized_phone:
            return {
                'numero': record_phone,
                'nome': str(record.get('nome', '')).strip(),
                'endereco': str(record.get('endereco', '')).strip(),
            }

    return None


def upsert_cliente(phone: str, nome: str, endereco: str) -> dict:
    normalized_phone = _normalize_phone(phone)
    worksheet = _get_worksheet(GOOGLE_SHEETS_CLIENTES_TAB)
    records = worksheet.get_all_records()

    for index, record in enumerate(records, start=2):
        record_phone = _normalize_phone(str(record.get('numero', '')))
        if record_phone == normalized_phone:
            worksheet.update(f'A{index}:C{index}', [[normalized_phone, nome.strip(), endereco.strip()]])
            return {
                'numero': normalized_phone,
                'nome': nome.strip(),
                'endereco': endereco.strip(),
            }

    worksheet.append_row([normalized_phone, nome.strip(), endereco.strip()])
    return {
        'numero': normalized_phone,
        'nome': nome.strip(),
        'endereco': endereco.strip(),
    }


def list_estoque() -> dict[str, dict]:
    worksheet = _get_worksheet(GOOGLE_SHEETS_ESTOQUE_TAB)
    records = worksheet.get_all_records()

    categories: dict[str, dict] = {}
    for record in records:
        category_name = str(record.get('categoria', '')).strip() or 'Sem categoria'
        parent_code = str(record.get('codigo_pai', '')).strip()
        name = str(record.get('nome_produto', '')).strip()
        variation_name = str(record.get('variacao', '')).strip()
        image_url = str(record.get('imagem_url', '')).strip()
        price_raw = record.get('preco', '')
        stock_raw = str(record.get('estoque', '')).strip()

        if not parent_code or not name:
            continue

        try:
            price = _parse_brl_price(price_raw)
        except ValueError:
            continue

        try:
            stock = int(float(stock_raw)) if stock_raw else 0
        except ValueError:
            stock = 0

        if stock <= 0:
            continue

        category_key = category_name.lower()
        category_group = categories.get(category_key)
        if not category_group:
            category_group = {
                'categoria': category_name,
                'produtos': {},
            }
            categories[category_key] = category_group

        products = category_group['produtos']
        product_key = parent_code.lower()
        product_group = products.get(product_key)
        if not product_group:
            product_group = {
                'codigo_pai': parent_code,
                'nome_produto': name,
                'imagem_url': image_url,
                'variacoes': [],
            }
            products[product_key] = product_group

        if image_url and not product_group.get('imagem_url'):
            product_group['imagem_url'] = image_url

        product_group['variacoes'].append(
            {
                'variacao': variation_name or 'Unico',
                'preco': price,
                'estoque': stock,
            }
        )

    return categories


def get_ultimo_pedido(phone: str) -> dict | None:
    normalized_phone = _normalize_phone(phone)
    if not normalized_phone:
        return None

    worksheet = _get_worksheet(GOOGLE_SHEETS_PEDIDOS_TAB)
    records = worksheet.get_all_records()

    for record in reversed(records):
        record_phone = _normalize_phone(str(record.get('numero', '')))
        if record_phone == normalized_phone:
            return {
                'numero': record_phone,
                'status': str(record.get('status', '')).strip(),
                'itens_resumo': str(record.get('itens_resumo', '')).strip(),
                'criado_em': str(record.get('criado_em', '')).strip(),
            }

    return None


def create_pedido(
    *,
    phone: str,
    nome: str,
    endereco: str,
    itens_resumo: str,
    total: float,
    forma_pagamento: str,
    status: str = 'novo',
) -> dict:
    worksheet = _get_worksheet(GOOGLE_SHEETS_PEDIDOS_TAB)
    created_at = datetime.now(timezone.utc).isoformat()
    normalized_phone = _normalize_phone(phone)

    row = [
        created_at,
        normalized_phone,
        nome.strip(),
        endereco.strip(),
        itens_resumo.strip(),
        f'{total:.2f}',
        forma_pagamento.strip(),
        status,
    ]
    worksheet.append_row(row)

    return {
        'criado_em': created_at,
        'numero': normalized_phone,
        'nome': nome.strip(),
        'endereco': endereco.strip(),
        'itens_resumo': itens_resumo.strip(),
        'total': total,
        'forma_pagamento': forma_pagamento.strip(),
        'status': status,
    }
