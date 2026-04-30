from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from decimal import Decimal
from typing import Any

import asyncpg
import requests

from config import (
    BOT_ESTOQUE_DEFAULT_CATEGORY,
    BOT_DATABASE_CONNECTION_URI,
)


logger = logging.getLogger(__name__)

NEXT_REVALIDATE_URL = 'http://whatsapp_saas_web:3000/api/revalidate'


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

    # Test mode: pizzaria/adega/lanchonete sem produtos reais
    if normalized_tenant_id in ['pizzaria', 'adega', 'lanchonete']:
        logger.warning(f"[FALLBACK] fetch_active_produtos em test mode para {normalized_tenant_id}")
        return []

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

        if rows:
            return [dict(row) for row in rows]

        # Fallback: quando o catálogo híbrido ainda não foi populado,
        # usamos os itens ativos do estoque para manter o bot alinhado com a tela de Estoque.
        stock_rows = await conn.fetch(
            '''
            SELECT
                id,
                nome,
                variacao,
                preco,
                quantidade
            FROM stock_items
            WHERE tenant_id = $1
              AND quantidade > 0
            ORDER BY nome ASC
            ''',
            normalized_tenant_id,
        )
    finally:
        await conn.close()

    mapped_rows: list[dict[str, Any]] = []
    for row in stock_rows:
        mapped_rows.append(
            {
                'id': str(row.get('id')),
                'nome': str(row.get('nome') or '').strip(),
                'categoria': str(row.get('variacao') or BOT_ESTOQUE_DEFAULT_CATEGORY or 'Sem categoria').strip(),
                'preco_base': _to_float(row.get('preco')),
                'classe_negocio': 'delivery',
                'config_nicho': {},
                'regras_ia': 'Produto carregado automaticamente do estoque ativo.',
            }
        )

    return mapped_rows


async def get_tenant_configs(tenant_id: str) -> dict[str, Any]:
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        raise ValueError('tenant_id é obrigatório para carregar configurações.')

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    
    row = None
    try:
        row = await conn.fetchrow(
            '''
            SELECT
                "promptIa" AS prompt_ia,
                "whatsappAdmin" AS whatsapp_admin,
                "botObjective" AS bot_objective,
                "companyName" AS nome_negocio,
                "botName" AS nome_atendente,
                config_nicho
            FROM tenants
            WHERE id = $1
            LIMIT 1
            ''',
            normalized_tenant_id,
        )
    except Exception as e:
        # Fallback: se não conseguir fazer query (UUID inválido), tentar com sub_nicho
        if normalized_tenant_id in ['pizzaria', 'adega', 'lanchonete']:
            logger.warning(f"[FALLBACK] tenant_id '{normalized_tenant_id}' não é UUID, usando como test mode")
        else:
            raise
    finally:
        await conn.close()

    prompt_ia = ''
    whatsapp_admin = ''
    bot_objective = ''
    nome_negocio = ''
    nome_atendente = ''
    config_nicho: dict[str, Any] = {}

    tenant: dict[str, Any] = {}
    if row:
        tenant = dict(row)
        prompt_ia = str(row.get('prompt_ia') or '').strip()
        whatsapp_admin = str(row.get('whatsapp_admin') or '').strip()
        bot_objective = str(row.get('bot_objective') or '').strip().upper()
        nome_negocio = str(row.get('nome_negocio') or '').strip()
        nome_atendente = str(row.get('nome_atendente') or '').strip()

        raw_cfg = row.get('config_nicho')
        if isinstance(raw_cfg, str):
            try:
                parsed_cfg = json.loads(raw_cfg)
                if isinstance(parsed_cfg, dict):
                    config_nicho = parsed_cfg
            except Exception as e:
                logger.error(f"[TENANT] Failed to parse JSON config_nicho: {e}")
                config_nicho = {}
        elif isinstance(raw_cfg, dict):
            config_nicho = raw_cfg

    # Fallback: se row for None (test mode com pizzaria/adega/lanchonete), usar sub_nicho como config
    if not row and normalized_tenant_id in ['pizzaria', 'adega', 'lanchonete']:
        nome_negocio = f'{normalized_tenant_id.title()}'
        config_nicho = {
            'sub_nicho': normalized_tenant_id,
            'faz_delivery': True,
            'horarios': [],
        }
        prompt_ia = ''  # Usará fallback de TEST_DRIVE_SUB_NICHO_RULES
        bot_objective = 'FECHAR_PEDIDO'

    sub_nicho = str(config_nicho.get('sub_nicho') or config_nicho.get('subNicho') or '').strip().lower()
    horarios = config_nicho.get('horarios')
    if not isinstance(horarios, list):
        horarios = []

    faz_delivery = bool(
        config_nicho.get('faz_delivery')
        or config_nicho.get('delivery')
        or config_nicho.get('accept_delivery')
    )
    faz_retirada = bool(
        config_nicho.get('faz_retirada')
        or config_nicho.get('retirada')
        or config_nicho.get('accept_pickup')
    )

    return {
        'promptIa': prompt_ia or DEFAULT_TENANT_PROMPT,
        'whatsappAdmin': whatsapp_admin,
        'botObjective': bot_objective or 'FECHAR_PEDIDO',
        'nome_negocio': nome_negocio or 'nossa loja',
        'nome_atendente': nome_atendente or (nome_negocio.split()[0] if nome_negocio else 'Atendente'),
        'sub_nicho': sub_nicho or 'adega',
        'horarios': horarios,
        'faz_delivery': faz_delivery,
        'faz_retirada': faz_retirada,
        'config_nicho': config_nicho,
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


async def fetch_stock_for_context(tenant_id: str) -> dict:
    """
    Retorna dados de cardápio para montar o contexto do bot por sub-nicho.

    Formato de retorno:
    {
      "sub_nicho": str,
      "items": [{"nome": str, "preco": float, "quantidade": int, "disponivel": bool}],
      "combos": [{"nome": str, "descricao": str, "preco": float, "disponivel": bool}],
      "sabores": [{"nome": str, "categoria": str, "disponivel": bool, "precos": {"P": 35.0}}],
      "tamanhos": [{"sigla": str, "nome": str, "fatias": int, "modificador_preco": float}],
      "bordas": [{"nome": str, "preco_extra": float}],
    }
    """
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        tenant_row = await conn.fetchrow(
            'SELECT config_nicho FROM tenants WHERE id = $1 LIMIT 1',
            tenant_id,
        )

        sub_nicho = None
        if tenant_row:
            raw_cfg = tenant_row['config_nicho']
            if isinstance(raw_cfg, str):
                try:
                    cfg = json.loads(raw_cfg)
                except Exception:
                    cfg = {}
            elif isinstance(raw_cfg, dict):
                cfg = raw_cfg
            else:
                cfg = {}

            if isinstance(cfg, dict):
                sub_nicho = cfg.get('sub_nicho') or cfg.get('subNicho')
        
        logger.info(f"[FETCH-STOCK] tenant_id: {tenant_id} | sub_nicho: {sub_nicho}")

        if sub_nicho == 'pizzaria':
            logger.info(f"[FETCH-STOCK] Iniciando queries para pizzaria...")
            try:
                # Fazer as queries sequencialmente, não em paralelo
                sabor_rows = await conn.fetch(
                    '''
                    SELECT nome, variacao, preco, ativo
                    FROM stock_items
                    WHERE tenant_id = $1
                      AND ativo = TRUE
                    ORDER BY nome ASC
                    ''',
                    tenant_id,
                )
                logger.info(f"[FETCH-STOCK] Sabores retornados: {len(sabor_rows)}")
                
                tamanho_rows = await conn.fetch(
                    '''
                    SELECT sigla, nome, fatias, modificador_preco
                    FROM pizza_tamanhos
                    WHERE tenant_id = $1
                    ORDER BY ordem ASC, created_at ASC
                    ''',
                    tenant_id,
                )
                logger.info(f"[FETCH-STOCK] Tamanhos retornados: {len(tamanho_rows)}")
                
                borda_rows = await conn.fetch(
                    '''
                    SELECT nome, preco_extra
                    FROM pizza_bordas
                    WHERE tenant_id = $1
                    ORDER BY created_at ASC
                    ''',
                    tenant_id,
                )
                logger.info(f"[FETCH-STOCK] Bordas retornadas: {len(borda_rows)}")

                bebida_rows = []
                try:
                    bebida_rows = await conn.fetch(
                        '''
                        SELECT nome, preco, ativo
                        FROM bebidas
                        WHERE tenant_id = $1
                        ORDER BY created_at ASC
                        ''',
                        tenant_id,
                    )
                except Exception as bebida_error:
                    logger.warning(f"[FETCH-STOCK] Query de bebidas indisponivel: {bebida_error}")
                logger.info(f"[FETCH-STOCK] Bebidas retornadas: {len(bebida_rows)}")
                
            except Exception as e:
                logger.error(f"[FETCH-STOCK] Erro nas queries: {e}", exc_info=True)
                raise

            tamanhos: list[dict[str, Any]] = []
            for row in tamanho_rows:
                tamanhos.append(
                    {
                        'sigla': str(row.get('sigla') or '').strip(),
                        'nome': str(row.get('nome') or '').strip(),
                        'fatias': int(row.get('fatias') or 0),
                        'modificador_preco': _to_float(row.get('modificador_preco') or 0),
                    }
                )

            sabores: list[dict[str, Any]] = []
            for row in sabor_rows:
                base = _to_float(row.get('preco') or 0)
                precos: dict[str, float] = {}
                for tamanho in tamanhos:
                    sigla = str(tamanho.get('sigla') or '').strip()
                    if not sigla:
                        continue
                    precos[sigla] = base + float(tamanho.get('modificador_preco') or 0)

                sabores.append(
                    {
                        'nome': str(row.get('nome') or '').strip(),
                        'categoria': str(row.get('variacao') or 'Outros').strip(),
                        'disponivel': bool(row.get('ativo')),
                        'precos': precos,
                    }
                )

            bordas: list[dict[str, Any]] = []
            for row in borda_rows:
                bordas.append(
                    {
                        'nome': str(row.get('nome') or '').strip(),
                        'preco_extra': _to_float(row.get('preco_extra') or 0),
                    }
                )

            bebidas: list[dict[str, Any]] = []
            for row in bebida_rows:
                bebidas.append(
                    {
                        'nome': str(row.get('nome') or '').strip(),
                        'preco': _to_float(row.get('preco') or 0),
                        'disponivel': bool(row.get('ativo')),
                    }
                )

            if not bebidas:
                bebidas = [
                    {'nome': 'Coca-Cola 600ml', 'preco': 6.0, 'disponivel': True},
                    {'nome': 'Coca-Cola 2L', 'preco': 12.0, 'disponivel': True},
                    {'nome': 'Guarana Antarctica 600ml', 'preco': 5.0, 'disponivel': True},
                    {'nome': 'Guarana Antarctica 2L', 'preco': 10.0, 'disponivel': True},
                    {'nome': 'Agua mineral 500ml', 'preco': 3.0, 'disponivel': True},
                ]

            logger.info(f"[FETCH-STOCK-PIZZA] Retornando: {len(sabores)} sabores, {len(tamanhos)} tamanhos, {len(bordas)} bordas, {len(bebidas)} bebidas")
            
            return {
                'sub_nicho': 'pizzaria',
                'sabores': sabores,
                'tamanhos': tamanhos,
                'bordas': bordas,
                'bebidas': bebidas,
            }

        item_rows = await conn.fetch(
            '''
            SELECT id, nome, preco, variacao, tem_variacoes, quantidade, ativo
            FROM stock_items
            WHERE tenant_id = $1
            ORDER BY nome ASC
            ''',
            tenant_id,
        )

        variacao_rows = []
        adicional_rows = []
        if sub_nicho == 'lanchonete':
            variacao_rows = await conn.fetch(
                '''
                SELECT item_id, sigla, nome, preco, ordem
                FROM item_variacoes
                WHERE item_id IN (
                    SELECT id FROM stock_items WHERE tenant_id = $1
                )
                ORDER BY ordem ASC, created_at ASC
                ''',
                tenant_id,
            )
            adicional_rows = await conn.fetch(
                '''
                SELECT item_id, nome, preco_extra
                FROM item_adicionais
                WHERE ativo = TRUE
                    AND item_id IN (
                        SELECT id FROM stock_items WHERE tenant_id = $1
                    )
                ORDER BY created_at ASC
                ''',
                tenant_id,
            )

        combo_rows = []
        if sub_nicho == 'lanchonete':
            combo_rows = await conn.fetch(
                '''
                SELECT nome, descricao, preco, ativo
                FROM combos
                WHERE tenant_id = $1
                  AND ativo = TRUE
                ORDER BY nome ASC
                ''',
                tenant_id,
            )
        else:
            combo_rows = await conn.fetch(
                '''
                SELECT nome, preco_base, config_nicho
                FROM produtos
                WHERE tenant_id = $1
                  AND ativo = TRUE
                  AND classe_negocio = \'combo\'
                ORDER BY nome ASC
                ''',
                tenant_id,
            )
    finally:
        await conn.close()

    import json as _json

    stock_map: dict[str, int] = {}
    variacoes_por_item: dict[str, list[dict]] = {}
    adicionais_por_item: dict[str, list[dict]] = {}
    for row in variacao_rows:
        item_id = str(row.get('item_id') or '').strip()
        if not item_id:
            continue
        variacoes_por_item.setdefault(item_id, []).append({
            'sigla': str(row.get('sigla') or '').strip(),
            'nome': str(row.get('nome') or '').strip(),
            'preco': _to_float(row.get('preco') or 0),
        })

    for row in adicional_rows:
        item_id = str(row.get('item_id') or '').strip()
        if not item_id:
            continue
        adicionais_por_item.setdefault(item_id, []).append({
            'nome': str(row.get('nome') or '').strip(),
            'preco_extra': _to_float(row.get('preco_extra') or 0),
        })

    items: list[dict] = []
    for row in item_rows:
        item_id = str(row['id'] or '').strip()
        nome = str(row['nome'] or '').strip()
        qty = int(row['quantidade'] or 0)
        ativo = bool(row.get('ativo'))
        disponivel = ativo if sub_nicho == 'lanchonete' else qty > 0
        stock_map[nome.lower()] = qty
        items.append({
            'nome': nome,
            'preco': _to_float(row['preco']),
            'categoria': str(row.get('variacao') or 'Outros').strip() or 'Outros',
            'quantidade': qty,
            'disponivel': disponivel,
            'tem_variacoes': bool(row.get('tem_variacoes')),
            'variacoes': variacoes_por_item.get(item_id, []),
            'adicionais': adicionais_por_item.get(item_id, []),
        })

    combos: list[dict] = []
    for row in combo_rows:
        nome = str(row['nome'] or '').strip()

        if 'preco' in row.keys():
            combos.append(
                {
                    'nome': nome,
                    'descricao': str(row.get('descricao') or '').strip(),
                    'preco': _to_float(row.get('preco') or 0),
                    'disponivel': bool(row.get('ativo')),
                }
            )
            continue

        raw_cfg = row['config_nicho']
        if isinstance(raw_cfg, str):
            try:
                cfg = _json.loads(raw_cfg)
            except Exception:
                cfg = {}
        elif isinstance(raw_cfg, dict):
            cfg = raw_cfg
        else:
            cfg = {}

        componentes: list[str] = cfg.get('componentes', []) if isinstance(cfg, dict) else []
        disponivel = (
            all(stock_map.get(c.lower(), 0) > 0 for c in componentes)
            if componentes
            else True
        )
        descricao = cfg.get('descricao', ' + '.join(componentes)) if isinstance(cfg, dict) else ' + '.join(componentes)
        combos.append({
            'nome': nome,
            'descricao': descricao,
            'preco': _to_float(row['preco_base']),
            'disponivel': disponivel,
        })

    return {'sub_nicho': sub_nicho or 'adega', 'items': items, 'combos': combos}


async def get_tenant_sub_nicho(tenant_id: str) -> str | None:
    """Retorna o sub_nicho do tenant (ex: 'adega', 'pizzaria') ou None."""
    if not BOT_DATABASE_CONNECTION_URI:
        raise RuntimeError('DATABASE_CONNECTION_URI não configurada.')

    normalized_tenant_id = (tenant_id or '').strip()
    
    # Test mode: return sub_nicho directly if it's a known niche name
    if normalized_tenant_id in ['pizzaria', 'adega', 'lanchonete']:
        return normalized_tenant_id

    conn = await asyncpg.connect(BOT_DATABASE_CONNECTION_URI)
    try:
        row = await conn.fetchrow(
            'SELECT config_nicho FROM tenants WHERE id = $1 LIMIT 1',
            normalized_tenant_id,
        )
    except Exception:
        # If query fails due to invalid UUID, return None (no specific sub_nicho)
        return None
    finally:
        await conn.close()

    if not row:
        return None

    import json as _json

    raw_cfg = row['config_nicho']
    if isinstance(raw_cfg, str):
        try:
            cfg = _json.loads(raw_cfg)
        except Exception:
            return None
    elif isinstance(raw_cfg, dict):
        cfg = raw_cfg
    else:
        return None

    if not isinstance(cfg, dict):
        return None

    value = cfg.get('sub_nicho') or cfg.get('subNicho')
    if isinstance(value, str):
        return value

    return None


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
                new_customer_id = str(uuid.uuid4())
                inserted_customer = await conn.fetchrow(
                    '''
                    INSERT INTO customers (id, telefone, nome, endereco, tenant_id)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    ''',
                    new_customer_id,
                    normalized_phone,
                    customer_name,
                    customer_address,
                    normalized_tenant_id,
                )
                customer_id = str(inserted_customer.get('id'))

            # B) Insert do Order
            inserted_order = await conn.fetchrow(
                '''
                INSERT INTO orders (id, status, total, forma_pagamento, tenant_id, customer_id)
                VALUES ($1, 'NOVO', $2, $3, $4, $5)
                RETURNING id, status, data_criacao
                ''',
                str(uuid.uuid4()),
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
                    INSERT INTO order_items (id, order_id, nome_produto, quantidade, preco_unitario)
                    VALUES ($1, $2, $3, $4, $5)
                    ''',
                    str(uuid.uuid4()),
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

    try:
        requests.post(
            NEXT_REVALIDATE_URL,
            json={'tag': 'pedidos', 'tenant_id': normalized_tenant_id},
            timeout=3,
        )
    except Exception:
        pass

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


async def save_pizza_order(tenant_id: str, phone: str, session: dict[str, Any]) -> dict[str, Any] | None:
    """Persiste pedido finalizado do fluxo de pizzaria no mesmo pipeline de pedidos padrão."""
    endereco = str(session.get('endereco') or '').strip()
    pagamento = str(session.get('pagamento') or '').strip()

    if not endereco or not pagamento:
        return None

    carrinho = session.get('carrinho', [])
    if not isinstance(carrinho, list):
        carrinho = []

    cart_items: list[dict[str, Any]] = []
    total = 0.0

    for pizza in carrinho:
        sabor = str((pizza or {}).get('sabor') or '').strip()
        tamanho = str((pizza or {}).get('tamanho') or '').strip().upper()
        borda = str((pizza or {}).get('borda') or 'Sem borda').strip()
        try:
            preco = float((pizza or {}).get('preco') or 0)
        except (TypeError, ValueError):
            preco = 0.0

        if not sabor:
            continue

        total += preco
        borda_str = f' + {borda}' if borda and borda.lower() != 'sem borda' else ''
        item_nome = f'{sabor} {tamanho}{borda_str}'.strip()

        cart_items.append(
            {
                'product_name': item_nome,
                'name': item_nome,
                'quantity': 1,
                'price': preco,
            }
        )

    bebidas_carrinho = session.get('bebidas_carrinho', [])
    if not isinstance(bebidas_carrinho, list):
        bebidas_carrinho = []

    for bebida in bebidas_carrinho:
        nome = str((bebida or {}).get('nome') or '').strip()
        if not nome:
            continue

        try:
            preco = float((bebida or {}).get('preco') or 0)
        except (TypeError, ValueError):
            preco = 0.0

        try:
            quantidade = int((bebida or {}).get('quantidade') or 1)
        except (TypeError, ValueError):
            quantidade = 1
        if quantidade <= 0:
            quantidade = 1

        total += preco * quantidade
        cart_items.append(
            {
                'product_name': nome,
                'name': nome,
                'quantity': quantidade,
                'price': preco,
            }
        )

    # Compatibilidade com sessões antigas sem carrinho.
    if not cart_items:
        sabor_legacy = str(session.get('sabor') or session.get('sabor_sugerido') or '').strip()
        tamanho_legacy = str(session.get('tamanho') or '').strip().upper()
        borda_legacy = str(session.get('borda') or 'Sem borda').strip()
        if not sabor_legacy:
            return None

        try:
            total_legacy = float(session.get('total') or 0)
        except (TypeError, ValueError):
            total_legacy = 0.0

        borda_str = f' + {borda_legacy}' if borda_legacy and borda_legacy.lower() != 'sem borda' else ''
        item_nome = f'{sabor_legacy} {tamanho_legacy}{borda_str}'.strip()
        cart_items.append(
            {
                'product_name': item_nome,
                'name': item_nome,
                'quantity': 1,
                'price': total_legacy,
            }
        )
        total = total_legacy

    try:
        total_from_session = float(session.get('total') or 0)
    except (TypeError, ValueError):
        total_from_session = 0.0
    if total <= 0 and total_from_session > 0:
        total = total_from_session

    return await save_order(
        tenant_id=tenant_id,
        phone=phone,
        nome='Cliente WhatsApp',
        endereco=endereco,
        cart_items=cart_items,
        total=total,
        forma_pagamento=pagamento,
    )