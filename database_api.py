from __future__ import annotations

import asyncio
import re
from decimal import Decimal
from typing import Any

import asyncpg

from config import (
    BOT_ESTOQUE_DEFAULT_CATEGORY,
    BOT_DATABASE_CONNECTION_URI,
)


DEFAULT_TENANT_PROMPT = (
    'Você é um atendente virtual de loja, cordial e objetivo. '
    'Responda de forma clara, educada e sempre focada em ajudar o cliente a concluir a compra. '
    'Nunca invente preços, produtos ou condições de pagamento.'
)


def _slugify(value: str) -> str:
    normalized = (value or '').strip().lower()
    normalized = re.sub(r'\s+', '-', normalized)
    normalized = re.sub(r'[^a-z0-9\-]', '', normalized)
    return normalized or 'produto-sem-codigo'


def _to_float(value) -> float:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


async def _fetch_stock_rows(tenant_id: str) -> list[asyncpg.Record]:
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        return await conn.fetch(
            '''
            SELECT
                id,
                nome,
                variacao,
                preco,
                quantidade,
                data_criacao
            FROM stock_items
            WHERE tenant_id = $1
            ORDER BY nome ASC, data_criacao ASC
            ''',
            tenant_id,
        )
    finally:
        await conn.close()


async def fetch_active_produtos(tenant_id: str) -> list[dict[str, Any]]:
    """Busca produtos ativos do catálogo híbrido para um tenant."""
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        raise ValueError('tenant_id é obrigatório para buscar produtos ativos.')

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        rows = await conn.fetch(
            '''
            SELECT
                id,
                nome,
                categoria,
                preco_base,
                classe_negocio,
                config_nicho,
                regras_ia
            FROM produtos
            WHERE tenant_id = $1
              AND ativo = TRUE
            ORDER BY categoria ASC, nome ASC
            ''',
            normalized_tenant_id,
        )
    finally:
        await conn.close()

    return [dict(row) for row in rows]


async def get_tenant_configs(tenant_id: str) -> dict[str, str]:
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        raise ValueError('tenant_id é obrigatório para carregar configurações.')

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        row = await conn.fetchrow(
            '''
            SELECT
                "promptIa" AS prompt_ia,
                "whatsappAdmin" AS whatsapp_admin,
                "botObjective" AS bot_objective
            FROM tenants
            WHERE id = $1
            LIMIT 1
            ''',
            normalized_tenant_id,
        )
    finally:
        await conn.close()

    prompt_ia = ''
    whatsapp_admin = ''
    bot_objective = ''

    if row:
        prompt_ia = str(row.get('prompt_ia') or '').strip()
        whatsapp_admin = str(row.get('whatsapp_admin') or '').strip()
        bot_objective = str(row.get('bot_objective') or '').strip().upper()

    return {
        'promptIa': prompt_ia or DEFAULT_TENANT_PROMPT,
        'whatsappAdmin': whatsapp_admin,
        'botObjective': bot_objective or 'FECHAR_PEDIDO',
    }


async def get_tenant_by_instance(instance_name: str) -> str | None:
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_instance_name = (instance_name or '').strip()
    if not normalized_instance_name:
        return None

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        row = await conn.fetchrow(
            '''
            SELECT id
            FROM tenants
            WHERE "evolutionInstanceName" = $1
            LIMIT 1
            ''',
            normalized_instance_name,
        )
    finally:
        await conn.close()

    if not row:
        return None

    tenant_id = str(row.get('id') or '').strip()
    return tenant_id or None


def _normalize_phone(value: str) -> str:
    return re.sub(r'\D', '', value or '')


async def _fetch_customer_by_phone(tenant_id: str, phone: str) -> dict[str, str] | None:
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_phone = _normalize_phone(phone)
    if not normalized_phone:
        return None

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        row = await conn.fetchrow(
            '''
            SELECT telefone, nome, endereco
            FROM customers
            WHERE tenant_id = $1
              AND regexp_replace(telefone, '\\D', '', 'g') = $2
            ORDER BY nome ASC
            LIMIT 1
            ''',
            tenant_id,
            normalized_phone,
        )
    finally:
        await conn.close()

    if not row:
        return None

    return {
        'numero': _normalize_phone(str(row.get('telefone') or '')),
        'nome': str(row.get('nome') or '').strip(),
        'endereco': str(row.get('endereco') or '').strip(),
    }


async def _fetch_last_order_by_phone(tenant_id: str, phone: str) -> dict[str, str] | None:
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_phone = _normalize_phone(phone)
    if not normalized_phone:
        return None

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        order_row = await conn.fetchrow(
            '''
            SELECT
                o.id,
                o.status,
                o.data_criacao,
                c.telefone
            FROM orders o
            JOIN customers c ON c.id = o.customer_id
            WHERE o.tenant_id = $1
              AND regexp_replace(c.telefone, '\\D', '', 'g') = $2
            ORDER BY o.data_criacao DESC
            LIMIT 1
            ''',
            tenant_id,
            normalized_phone,
        )

        if not order_row:
            return None

        items = await conn.fetch(
            '''
            SELECT nome_produto, quantidade
            FROM order_items
            WHERE order_id = $1
            ORDER BY id ASC
            ''',
            str(order_row.get('id')),
        )
    finally:
        await conn.close()

    itens_resumo = '\n'.join(
        f"{int(item.get('quantidade') or 0)}x {str(item.get('nome_produto') or '').strip()}"
        for item in items
        if str(item.get('nome_produto') or '').strip()
    ).strip()

    return {
        'numero': _normalize_phone(str(order_row.get('telefone') or '')),
        'status': str(order_row.get('status') or '').strip(),
        'itens_resumo': itens_resumo or 'Itens não informados',
        'criado_em': str(order_row.get('data_criacao') or ''),
    }


async def create_produto(
    tenant_id: str,
    nome: str,
    categoria: str,
    preco_base: float,
    classe_negocio: str = 'generico',
    regras_ia: str | None = None,
    config_nicho: dict | None = None,
) -> str | None:
    """Cria um novo produto no catálogo híbrido e retorna o ID do produto criado."""
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_tenant_id = (tenant_id or '').strip()
    normalized_nome = (nome or '').strip()
    normalized_categoria = (categoria or '').strip()
    normalized_classe_negocio = (classe_negocio or 'generico').strip()
    normalized_regras_ia = (regras_ia or '').strip() or None

    if not normalized_tenant_id or not normalized_nome or not normalized_categoria:
        raise ValueError('tenant_id, nome e categoria são obrigatórios para criar um produto.')

    try:
        preco_base_float = float(preco_base)
    except (TypeError, ValueError):
        raise ValueError(f'preco_base inválido: {preco_base}')

    if config_nicho is None:
        config_nicho = {}

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        result = await conn.fetchval(
            '''
            INSERT INTO produtos
            (tenant_id, nome, categoria, preco_base, classe_negocio, config_nicho, regras_ia, ativo)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, TRUE)
            RETURNING id
            ''',
            normalized_tenant_id,
            normalized_nome,
            normalized_categoria,
            preco_base_float,
            normalized_classe_negocio,
            config_nicho,
            normalized_regras_ia,
        )
        return result
    finally:
        await conn.close()


def list_estoque(tenant_id: str) -> dict[str, dict]:
    """
    Retorna o catálogo no mesmo formato usado hoje pelo message_buffer.py:
    {
      "categoria_key": {
        "categoria": str,
        "produtos": {
          "codigo_pai_key": {
            "codigo_pai": str,
            "nome_produto": str,
            "imagem_url": str,
            "variacoes": [
              {"variacao": str, "preco": float, "estoque": int}
            ]
          }
        }
      }
    }
    """
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        raise RuntimeError('tenant_id é obrigatório para listar estoque.')

    rows = asyncio.run(_fetch_stock_rows(normalized_tenant_id))

    category_name = (BOT_ESTOQUE_DEFAULT_CATEGORY or 'Sem categoria').strip()
    category_key = category_name.lower()
    categories: dict[str, dict] = {
        category_key: {
            'categoria': category_name,
            'produtos': {},
        }
    }

    products = categories[category_key]['produtos']

    for row in rows:
        nome_produto = str(row.get('nome') or '').strip()
        if not nome_produto:
            continue

        estoque_raw = row.get('quantidade')
        try:
            estoque = int(estoque_raw) if estoque_raw is not None else 0
        except (TypeError, ValueError):
            estoque = 0

        if estoque <= 0:
            continue

        try:
            preco = _to_float(row.get('preco'))
        except (TypeError, ValueError):
            continue

        variacao = str(row.get('variacao') or '').strip() or 'Unico'

        # Como o schema atual não possui codigo_pai/categoria/imagem_url,
        # usamos o nome do produto como agrupador (codigo_pai sintético).
        codigo_pai = _slugify(nome_produto)
        product_key = codigo_pai.lower()

        product_group = products.get(product_key)
        if not product_group:
            product_group = {
                'codigo_pai': codigo_pai,
                'nome_produto': nome_produto,
                'imagem_url': '',
                'variacoes': [],
            }
            products[product_key] = product_group

        product_group['variacoes'].append(
            {
                'variacao': variacao,
                'preco': preco,
                'estoque': estoque,
            }
        )

    return categories


def get_cliente_by_phone(phone: str, tenant_id: str) -> dict[str, str] | None:
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        raise RuntimeError('tenant_id é obrigatório para buscar cliente.')
    return asyncio.run(_fetch_customer_by_phone(normalized_tenant_id, phone))


def get_ultimo_pedido(phone: str, tenant_id: str) -> dict[str, str] | None:
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        raise RuntimeError('tenant_id é obrigatório para buscar último pedido.')
    return asyncio.run(_fetch_last_order_by_phone(normalized_tenant_id, phone))


async def save_order(
    tenant_id: str,
    phone: str,
    nome: str,
    endereco: str,
    cart_items: list[dict],
    total: float,
    forma_pagamento: str,
) -> dict[str, Any]:
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        raise RuntimeError('tenant_id é obrigatório para salvar pedido.')

    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_phone = _normalize_phone(phone)
    if not normalized_phone:
        raise ValueError('Telefone inválido para salvar pedido.')

    customer_name = (nome or '').strip() or 'Cliente'
    customer_address = (endereco or '').strip() or 'Endereço não informado'
    payment_method = (forma_pagamento or '').strip() or 'Não informado'

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        async with conn.transaction():
            # A) Upsert manual do Customer por tenant + telefone
            customer = await conn.fetchrow(
                '''
                SELECT id
                FROM customers
                WHERE tenant_id = $1
                  AND regexp_replace(telefone, '\\D', '', 'g') = $2
                LIMIT 1
                ''',
                normalized_tenant_id,
                normalized_phone,
            )

            if customer:
                customer_id = str(customer.get('id'))
                await conn.execute(
                    '''
                    UPDATE customers
                    SET nome = $1,
                        endereco = $2,
                        telefone = $3
                    WHERE id = $4
                    ''',
                    customer_name,
                    customer_address,
                    normalized_phone,
                    customer_id,
                )
            else:
                inserted_customer = await conn.fetchrow(
                    '''
                    INSERT INTO customers (telefone, nome, endereco, tenant_id)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    ''',
                    normalized_phone,
                    customer_name,
                    customer_address,
                    normalized_tenant_id,
                )
                customer_id = str(inserted_customer.get('id'))

            # B) Insert do Order
            inserted_order = await conn.fetchrow(
                '''
                INSERT INTO orders (status, total, forma_pagamento, tenant_id, customer_id)
                VALUES ('NOVO', $1, $2, $3, $4)
                RETURNING id, status, data_criacao
                ''',
                float(total),
                payment_method,
                normalized_tenant_id,
                customer_id,
            )
            order_id = str(inserted_order.get('id'))

            # C) Insert dos itens
            for item in cart_items:
                nome_produto = str(item.get('product_name') or item.get('name') or '').strip()
                if not nome_produto:
                    continue

                quantidade = int(item.get('quantity') or 0)
                if quantidade <= 0:
                    continue

                preco_unitario = float(item.get('price') or 0)

                await conn.execute(
                    '''
                    INSERT INTO order_items (order_id, nome_produto, quantidade, preco_unitario)
                    VALUES ($1, $2, $3, $4)
                    ''',
                    order_id,
                    nome_produto,
                    quantidade,
                    preco_unitario,
                )
    finally:
        await conn.close()

    itens_resumo = '\n'.join(
        (
            f"{int(item.get('quantity') or 0)}x "
            f"{str(item.get('product_name') or item.get('name') or '').strip()}"
        )
        for item in cart_items
        if str(item.get('product_name') or item.get('name') or '').strip()
        and int(item.get('quantity') or 0) > 0
    ).strip()

    return {
        'id': order_id,
        'criado_em': str(inserted_order.get('data_criacao') or ''),
        'numero': normalized_phone,
        'nome': customer_name,
        'endereco': customer_address,
        'itens_resumo': itens_resumo,
        'total': float(total),
        'forma_pagamento': payment_method,
        'status': str(inserted_order.get('status') or 'NOVO'),
    }