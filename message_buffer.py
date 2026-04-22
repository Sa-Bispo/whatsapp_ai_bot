import asyncio
import json
import logging
import re
import time
from pathlib import Path
import redis.asyncio as redis

from collections import defaultdict

from config import (
    REDIS_URL,
    BUFFER_KEY_SUFIX,
    DEBOUNCE_SECONDS,
    BUFFER_TTL,
)
from evolution_api import send_whatsapp_message, send_whatsapp_presence
try:
    from evolution_api import send_whatsapp_image_file, send_whatsapp_media
except ImportError:
    def send_whatsapp_image_file(number, file_path, caption=''):
        return None

    def send_whatsapp_media(number, caption, media_url):
        return None
from database_api import (
    get_tenant_configs,
    get_cliente_by_phone,
    get_ultimo_pedido,
    list_estoque,
    save_order,
    save_pizza_order,
    fetch_stock_for_context,
)
from chains import generate_persona_response
from memory import clear_session_history, get_session_history
from order_extractor import (
    build_structured_order_payload,
    build_order_payload_from_history_window,
    build_order_payload_from_history_window_async,
)
from pizza_flow import process_pizza_message, PizzaState
from adega_flow import process_adega_message, AdegaState, save_adega_order_payload
from lanchonete_flow import process_lanchonete_message, LanchoneteState, save_lanchonete_order_payload
from router import detect_intent, is_within_hours
from script_responses import (
    resposta_saudacao,
    resposta_horario,
    resposta_fora_horario,
    resposta_entrega,
    resposta_status_pedido,
    resposta_cardapio,
    resposta_cancelamento_confirmacao,
)


redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
debounce_tasks = defaultdict(asyncio.Task)
logger = logging.getLogger(__name__)

FLOW_STATE_SUFFIX = '_flow_state'
FLOW_CONTEXT_SUFFIX = '_flow_context'

STATE_VERIFICACAO_INICIAL = 'VERIFICACAO_INICIAL'
STATE_MENU_INICIAL = 'MENU_INICIAL'
STATE_DUVIDAS_SUPLEMENTOS = 'DUVIDAS_SUPLEMENTOS'
STATE_CATALOGO = 'CATALOGO'
STATE_ESCOLHENDO_CATEGORIA = 'ESCOLHENDO_CATEGORIA'
STATE_ADICIONANDO_CARRINHO = 'ADICIONANDO_CARRINHO'

UPSELL_CODE = 'coq-01'
WELCOME_IMAGE_PATH = Path(__file__).resolve().parent / 'image' / 'image.png'
CATALOG_IMAGE_PATH = Path(__file__).resolve().parent / 'image' / 'suplemento.png'
QUICK_COMMANDS_TEXT = (
    'Comandos rápidos:\n'
    '🧾 carrinho → ver pedido\n'
    '⬅️ menu → voltar ao início\n'
    '✅ finalizar → concluir pedido'
)


def log(*args):
    print('[BUFFER]', *args)


def _state_key(chat_id: str) -> str:
    return f'{chat_id}{FLOW_STATE_SUFFIX}'


def _context_key(chat_id: str) -> str:
    return f'{chat_id}{FLOW_CONTEXT_SUFFIX}'


def get_store_info_text() -> str:
    return (
        '📍 *Informações da Loja*\n'
        'Endereço: Av. Vitalidade, 350 - Centro\n'
        'Horário: Segunda a Sábado, 08h às 20h'
    )


def get_whey_guidance_text() -> str:
    return (
        'Boa escolha! Whey é fundamental para a recuperação e ganho de massa. 💪🥩\n\n'
        'Nós trabalhamos com as melhores marcas, como Max Titanium, Integralmedica, Growth e Dux.\n\n'
        'Você já tem preferência por alguma marca específica ou pelo tipo do Whey '
        '(Concentrado, Isolado ou Hidrolisado)?\n\n'
        'Se estiver na dúvida, me avise que eu te mando os nossos 3 mais vendidos! 🏆'
    )


def get_catalog_text(grouped_products: dict[str, dict]) -> str:
    if not grouped_products:
        return '📦 No momento estamos sem itens disponíveis no catálogo.'

    lines = ['O que você precisa?\n']

    for item in grouped_products.values():
        lines.append(f"• {item['nome_produto']}")

    lines.append('')
    lines.append(QUICK_COMMANDS_TEXT)

    return '\n'.join(lines)


def _with_quick_commands(text: str) -> str:
    return f'{text}\n\n{QUICK_COMMANDS_TEXT}'


def _number_emoji(index: int) -> str:
    mapping = {
        0: '0️⃣',
        1: '1️⃣',
        2: '2️⃣',
        3: '3️⃣',
        4: '4️⃣',
        5: '5️⃣',
        6: '6️⃣',
        7: '7️⃣',
        8: '8️⃣',
        9: '9️⃣',
    }
    return mapping.get(index, str(index))


def get_categories_text(categories: list[dict]) -> str:
    lines = [
        '📚 *Categorias do Catálogo*',
        'Responda com o número da categoria desejada:',
    ]

    for index, category in enumerate(categories, start=1):
        lines.append(f"{_number_emoji(index)} - {category['categoria']}")

    lines.append('0️⃣ - Ver todos os produtos')
    lines.append('')
    lines.append(QUICK_COMMANDS_TEXT)
    return '\n'.join(lines)


def get_variation_options_text(product_group: dict) -> str:
    lines = [
        f"💊 *{product_group['nome_produto']}*",
        'Escolha a variação. Responda com o número:',
    ]
    for index, variation in enumerate(product_group.get('variacoes', []), start=1):
        lines.append(
            f"{index} - {variation['variacao']} ({format_brl(float(variation['preco']))})"
        )
    return '\n'.join(lines)


def get_upsell_prompt_text() -> str:
    return (
        '🔥 Aproveitando: quem compra suplementos geralmente leva uma Coqueteleira para os treinos.\n'
        'Deseja adicionar uma por *R$ 25,00*?\n'
        'Responda *SIM* ou *NÃO*.'
    )


def get_payment_prompt_text() -> str:
    return (
        '💳 Qual será a forma de pagamento na entrega?\n'
        'Responda com o número:\n'
        '1 - Pix\n'
        '2 - Cartão\n'
        '3 - Dinheiro'
    )


def parse_item_message(user_message: str):
    tokens = user_message.strip().replace(',', '.').split()
    if not tokens:
        return None, None

    code = tokens[0].strip().lower()

    quantity = 1
    if len(tokens) > 1:
        if not tokens[1].isdigit():
            return None, None
        quantity = int(tokens[1])

    if quantity <= 0:
        return None, None

    return code, quantity


def format_brl(value: float) -> str:
    return f'R$ {value:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def _first_token(value: str) -> str:
    normalized = (value or '').strip().lower()
    if not normalized:
        return ''
    return normalized.split()[0]


def _extract_numeric_choice(value: str) -> int | None:
    token = _first_token(value)
    if not token:
        return None

    digits = ''.join(ch for ch in token if ch.isdigit())
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def _parse_yes_no(value: str) -> str | None:
    normalized = _first_token(value)
    if normalized in {'sim', 's'}:
        return 'yes'
    if normalized in {'nao', 'não', 'n'}:
        return 'no'
    return None


def _is_valid_media_url(value: str) -> bool:
    text = (value or '').strip().lower()
    return text.startswith('http://') or text.startswith('https://')


def _payment_label(choice: str) -> str | None:
    mapping = {
        '1': 'Pix',
        '2': 'Cartao',
        '3': 'Dinheiro',
    }
    return mapping.get(choice.strip())


def _normalize_lookup(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', (value or '').lower()).strip()


def _build_price_lookup_from_grouped_products(grouped_products: dict[str, dict]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for category in grouped_products.values():
        for product in category.get('produtos', {}).values():
            product_name = str(product.get('nome_produto') or '').strip()
            if not product_name:
                continue

            variations = product.get('variacoes') or []
            min_price: float | None = None
            for variation in variations:
                try:
                    value = float(variation.get('preco') or 0)
                except (TypeError, ValueError):
                    continue
                if value <= 0:
                    continue
                if min_price is None or value < min_price:
                    min_price = value

            if min_price is not None:
                lookup[_normalize_lookup(product_name)] = float(min_price)
    return lookup


def _resolve_item_price_with_fallback(
    item: dict,
    price_lookup: dict[str, float],
) -> float:
    try:
        price = float(item.get('price') or 0)
    except (TypeError, ValueError):
        price = 0.0

    if price > 0:
        return price

    base_name = str(item.get('base_product_name') or '').strip()
    product_name = str(item.get('product_name') or item.get('name') or '').strip()
    normalized_candidates = [
        _normalize_lookup(base_name),
        _normalize_lookup(product_name),
    ]

    for candidate in normalized_candidates:
        if candidate and candidate in price_lookup:
            return float(price_lookup[candidate])

    return 0.0


def build_cart_summary(
    cart_items: list[dict],
    grouped_products: dict[str, dict] | None = None,
) -> tuple[str, float]:
    if not cart_items:
        return 'Carrinho vazio.', 0.0

    lines = []
    total = 0.0
    price_lookup = _build_price_lookup_from_grouped_products(grouped_products or {})

    for item in cart_items:
        price = _resolve_item_price_with_fallback(item, price_lookup)
        quantity = int(item.get('quantity', 0))
        line_total = price * quantity
        total += line_total

        product_name = item.get('product_name') or item.get('name') or 'Produto'
        variation_name = item.get('variation') or item.get('size') or ''
        display_name = f'{product_name} {variation_name}'.strip()

        lines.append(f'• {quantity}x {display_name} — {format_brl(line_total)}')

    return '\n'.join(lines), total


def build_checkout_summary(
    cart_items: list[dict],
    address: str,
    payment_method: str,
    nome_negocio: str = 'nossa loja',
    tenant_id: str | None = None,
) -> tuple[str | None, float]:
    grouped_products: dict[str, dict] = {}
    normalized_tenant_id = (tenant_id or '').strip()
    if normalized_tenant_id:
        try:
            grouped_products = list_estoque(normalized_tenant_id)
        except Exception as error:
            log(f'Falha no fallback de preço do estoque: {error}')

    items_text, total = build_cart_summary(cart_items, grouped_products=grouped_products)
    normalized_address = (address or '').strip()
    normalized_payment = (payment_method or '').strip()

    if _cart_items_are_invalid(cart_items):
        logger.warning('[ORDER] build_checkout_summary recebeu cart_items inválidos; não montando resumo final')
        return None, total

    endereco_ok = bool(normalized_address and '[' not in normalized_address and len(normalized_address) > 5)
    pagamento_ok = bool(normalized_payment and '[' not in normalized_payment and len(normalized_payment) > 2)
    if not endereco_ok or not pagamento_ok:
        return None, total

    text = (
        '✅ *Pedido anotado!*\n\n'
        f'{items_text}\n'
        f'💰 *Total: {format_brl(total)}*\n\n'
        f'📍 {normalized_address}\n'
        f'💳 {normalized_payment}\n\n'
        'Tô separando já! Em breve saiu 🚀'
    )
    return text, total


async def clear_order_context(chat_id: str):
    await redis_client.delete(_state_key(chat_id))
    await redis_client.delete(_context_key(chat_id))


async def clear_cart(chat_id: str):
    await redis_client.hdel(_context_key(chat_id), 'cart_items')


async def _set_state(chat_id: str, state: str):
    await redis_client.set(_state_key(chat_id), state)


async def _get_state(chat_id: str) -> str | None:
    return await redis_client.get(_state_key(chat_id))


async def _get_context(chat_id: str) -> dict:
    return await redis_client.hgetall(_context_key(chat_id))


async def _save_context(chat_id: str, data: dict):
    await redis_client.hset(_context_key(chat_id), mapping=data)


def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _extract_summary_field(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ''
    return (match.group(1) or '').strip(' \n\t*-_')


def _extract_order_summary_fields(reply_message: str) -> tuple[str, str, str]:
    items = _extract_summary_field(
        reply_message,
        r'(?:🛒\s*\*?itens:?\*?)\s*(.+?)(?=(?:📍|\*?endere|💳|\*?pagamento|$))',
    )
    address = _extract_summary_field(
        reply_message,
        r'(?:📍\s*\*?endere[cç]o:?\*?)\s*(.+?)(?=(?:💳|\*?pagamento|$))',
    )
    payment = _extract_summary_field(
        reply_message,
        r'(?:💳\s*\*?pagamento:?\*?)\s*(.+)',
    )

    if not items:
        bullet_lines = []
        for line in (reply_message or '').splitlines():
            clean = line.strip()
            if re.match(r'^(?:🍺|•)\s*\d+\s*[xX]\s+.+', clean):
                bullet_lines.append(re.sub(r'^(?:🍺|•)\s*', '', clean).strip())
        if bullet_lines:
            items = ' + '.join(bullet_lines)

    if not address:
        for line in (reply_message or '').splitlines():
            clean = line.strip()
            if clean.startswith('📍'):
                address = re.sub(r'^📍\s*(?:\*?endere[cç]o:?\*?)?\s*', '', clean, flags=re.IGNORECASE).strip()
                break

    if not payment:
        for line in (reply_message or '').splitlines():
            clean = line.strip()
            if clean.startswith('💳'):
                payment = re.sub(r'^💳\s*(?:\*?pagamento:?\*?)?\s*', '', clean, flags=re.IGNORECASE).strip()
                break

    payment = payment.replace('✅', '').strip()
    if '\n' in payment:
        payment = payment.splitlines()[0].strip()
    return items, address, payment


def _extract_name_from_text(text: str) -> str:
    patterns = (
        r'(?i)meu nome e\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
        r'(?i)pode colocar no nome de\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
        r'(?i)nome[:\-]?\s*([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
    )

    for pattern in patterns:
        match = re.search(pattern, text or '')
        if match:
            return (match.group(1) or '').strip()

    return ''


def _extract_address_from_text(text: str) -> str:
    if not text:
        return ''

    match = re.search(
        r'(?i)(?:moro na\s+)?((?:rua|r\.|avenida|av\.?|travessa|tv\.?|alameda|estrada|rodovia)\s+.+)',
        text,
    )
    if not match:
        return ''

    address = (match.group(1) or '').strip(' ,.-')
    address = re.sub(
        r'(?i)(?:[\.,;:\- ]+)?(?:vou pagar|pagamento|pago no|pagar no|pix|cart[aã]o|dinheiro).*$','',
        address,
    ).strip(' ,.-')
    return address


def _extract_payment_from_text(text: str) -> str:
    normalized = _normalize_text(text)
    if 'pix' in normalized:
        return 'Pix'
    if any(token in normalized for token in ('cartao', 'cartão', 'credito', 'crédito', 'debito', 'débito')):
        return 'Cartão'
    if 'dinheiro' in normalized:
        return 'Dinheiro'
    return ''


def _address_looks_broken(value: str) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return True

    return any(
        token in normalized
        for token in ('tamanho ', 'pagamento', 'pix', 'cartao', 'cartão', 'dinheiro')
    )


def _resolve_customer_data_from_session(
    session_key: str,
    fallback_address: str,
    fallback_payment: str,
) -> tuple[str, str, str]:
    history = get_session_history(session_key)
    human_texts = [str(msg.content).strip() for msg in history.messages if getattr(msg, 'type', '') == 'human']

    customer_name = ''
    customer_address = '' if _address_looks_broken(fallback_address) else fallback_address.strip()
    payment_method = fallback_payment.strip()

    for text in reversed(human_texts):
        if not customer_name:
            customer_name = _extract_name_from_text(text)
        if not customer_address:
            customer_address = _extract_address_from_text(text)
        if not payment_method:
            payment_method = _extract_payment_from_text(text)
        if customer_name and customer_address and payment_method:
            break

    return customer_name or 'Cliente WhatsApp', customer_address, payment_method


def _build_cart_items_from_summary(items_text: str) -> list[dict]:
    normalized_items = (items_text or '').strip()
    if not normalized_items:
        return []

    quantity_match = re.match(r'^\s*(\d+)\s*[xX]\s+(.+)$', normalized_items)
    price_match = re.search(r'R\$\s*([\d\.,]+)', normalized_items, flags=re.IGNORECASE)
    parsed_price = 0.0
    if price_match:
        raw_price = price_match.group(1).replace('.', '').replace(',', '.')
        try:
            parsed_price = float(raw_price)
        except ValueError:
            parsed_price = 0.0

    if quantity_match:
        quantity = int(quantity_match.group(1))
        product_name = quantity_match.group(2).strip()
    else:
        quantity = 1
        product_name = normalized_items

    # Remove sufixo de preço quando o item vier no formato "Produto — R$ 9,90".
    product_name = re.sub(
        r'\s*[—-]\s*R\$\s*[\d\.,]+.*$',
        '',
        product_name,
        flags=re.IGNORECASE,
    ).strip()

    return [
        {
            'product_name': product_name,
            'name': product_name,
            'quantity': max(1, quantity),
            'price': parsed_price,
        }
    ]


def _has_placeholder_text(value: str) -> bool:
    normalized = (value or '').strip().lower()
    if not normalized:
        return True
    return (
        '[' in normalized
        or 'itens informados pelo cliente' in normalized
        or 'endereço informado pelo cliente' in normalized
        or 'endereco informado pelo cliente' in normalized
        or 'pagamento informado pelo cliente' in normalized
        or 'produto confirmado' in normalized
    )


def _calculate_cart_total(cart_items: list[dict]) -> float:
    total = 0.0
    for item in cart_items:
        raw_price = item.get('price', 0)
        if isinstance(raw_price, str) and raw_price.strip().startswith('['):
            continue
        try:
            price = float(raw_price or 0)
        except (TypeError, ValueError):
            price = 0.0
        try:
            quantity = int(item.get('quantity') or 1)
        except (TypeError, ValueError):
            quantity = 1
        total += max(0.0, price) * max(1, quantity)
    return total


def _clean_product_name_for_lookup(value: str) -> str:
    cleaned = str(value or '').strip()
    cleaned = re.sub(r'\s*[—-]\s*R\$\s*[\d\.,]+.*$', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


async def _fill_cart_item_prices_from_stock(tenant_id: str, cart_items: list[dict]) -> list[dict]:
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id or not cart_items:
        return cart_items

    try:
        stock_data = await fetch_stock_for_context(normalized_tenant_id)
    except Exception as error:
        log(f'Falha ao buscar estoque para precificação do pedido: {error}')
        return cart_items

    stock_items = stock_data.get('items', []) if isinstance(stock_data, dict) else []
    if not stock_items:
        return cart_items

    price_lookup: dict[str, float] = {}
    for product in stock_items:
        product_name = _clean_product_name_for_lookup(str(product.get('nome') or '').strip())
        if not product_name:
            continue
        try:
            product_price = float(product.get('preco') or 0)
        except (TypeError, ValueError):
            continue
        if product_price <= 0:
            continue
        price_lookup[_normalize_lookup(product_name)] = product_price

    resolved_items: list[dict] = []
    for item in cart_items:
        new_item = dict(item)
        current_price = new_item.get('price', 0)
        try:
            numeric_price = float(current_price or 0)
        except (TypeError, ValueError):
            numeric_price = 0.0

        if numeric_price > 0:
            resolved_items.append(new_item)
            continue

        source_name = (
            str(new_item.get('base_product_name') or '').strip()
            or str(new_item.get('product_name') or '').strip()
            or str(new_item.get('name') or '').strip()
        )
        cleaned_name = _clean_product_name_for_lookup(source_name)
        normalized_name = _normalize_lookup(cleaned_name)
        resolved_price = 0.0

        if normalized_name in price_lookup:
            resolved_price = price_lookup[normalized_name]
        else:
            for candidate_name, candidate_price in price_lookup.items():
                if normalized_name and (candidate_name in normalized_name or normalized_name in candidate_name):
                    resolved_price = candidate_price
                    break

        if resolved_price > 0:
            new_item['price'] = resolved_price
            if cleaned_name:
                new_item['product_name'] = cleaned_name
                new_item['name'] = cleaned_name

        resolved_items.append(new_item)

    return resolved_items


def _cart_items_are_invalid(cart_items: list[dict]) -> bool:
    if not cart_items:
        return True

    for item in cart_items:
        name = str(item.get('product_name') or item.get('name') or '').strip()
        qty = int(item.get('quantity') or 0)
        if not name or qty <= 0:
            return True
        if _has_placeholder_text(name):
            return True

    return False


async def _persist_ai_final_order_if_needed(
    chat_id: str,
    tenant_id: str,
    reply_message: str,
) -> None:
    normalized_reply = (reply_message or '').strip()
    is_old_template = '*Resumo do Pedido*' in normalized_reply and normalized_reply.endswith('✅')
    is_new_template = normalized_reply.startswith('✅ *Pedido anotado!*')
    if not (is_old_template or is_new_template):
        return

    items_text, address, payment = _extract_order_summary_fields(normalized_reply)
    session_key = f'{tenant_id}:{chat_id}'
    structured_payload = await build_structured_order_payload(
        session_key=session_key,
        tenant_id=tenant_id,
        fallback_items_text=items_text,
        fallback_address=address,
        fallback_payment=payment,
    )
    order_data = structured_payload.get('order') or {}
    customer_name = str(order_data.get('customer_name') or 'Cliente WhatsApp').strip()
    resolved_address = str(order_data.get('customer_address') or '').strip()
    resolved_payment = str(order_data.get('payment_method') or '').strip()
    resolved_items_text = str(order_data.get('items_text') or '').strip()
    cart_items = [
        {
            'product_name': str(item.get('product_name') or '').strip(),
            'name': str(item.get('product_name') or '').strip(),
            'quantity': max(1, int(item.get('quantity') or 1)),
            'price': float(item.get('price') or 0),
        }
        for item in (order_data.get('items') or [])
        if str(item.get('product_name') or '').strip()
    ]

    effective_items_text = items_text or resolved_items_text

    if not cart_items and effective_items_text:
        cart_items = _build_cart_items_from_summary(effective_items_text)

    if not effective_items_text or not resolved_address or not resolved_payment:
        log('Resumo final detectado, mas campos obrigatórios incompletos para persistência.')
        return

    if _has_placeholder_text(effective_items_text) or _has_placeholder_text(resolved_address) or _has_placeholder_text(resolved_payment):
        logger.warning('[ORDER] Tentativa de salvar pedido com placeholders — abortando persistência')
        return

    context_key = _context_key(chat_id)
    fingerprint = f'{effective_items_text}|{resolved_address}|{resolved_payment}'.strip().lower()
    last_fingerprint = (await redis_client.hget(context_key, 'last_order_fingerprint') or '').strip().lower()
    if last_fingerprint and last_fingerprint == fingerprint:
        log('Pedido final já persistido anteriormente para este chat; ignorando duplicidade.')
        return

    if _cart_items_are_invalid(cart_items):
        logger.warning('[ORDER] Tentativa de salvar pedido com dados inválidos — abortando')
        return

    total_calculado = _calculate_cart_total(cart_items)
    if total_calculado == 0:
        cart_items = await _fill_cart_item_prices_from_stock(tenant_id, cart_items)
        total_calculado = _calculate_cart_total(cart_items)

    logger.info('[ORDER] total calculado: R$%.2f', total_calculado)

    try:
        await save_order(
            tenant_id=tenant_id,
            phone=chat_id,
            nome=customer_name,
            endereco=resolved_address,
            cart_items=cart_items,
            total=total_calculado,
            forma_pagamento=resolved_payment,
        )
        await redis_client.hset(
            context_key,
            mapping={
                'last_order_fingerprint': fingerprint,
                'last_structured_order_payload': json.dumps(structured_payload, ensure_ascii=False),
            },
        )
        await redis_client.delete(f'session:{tenant_id}:{chat_id}:produto_confirmado')
        await redis_client.setex(f'session:{tenant_id}:{chat_id}:last_order_time', 3600, str(time.time()))
        await clear_order_context(chat_id)
        clear_session_history(session_key)
        log(f'Pedido persistido no Kanban para {tenant_id}:{chat_id}')
    except Exception as error:
        log(f'Falha ao persistir pedido no Kanban: {error}')


async def _load_grouped_products(tenant_id: str) -> dict[str, dict]:
    try:
        grouped_products = await asyncio.to_thread(list_estoque, tenant_id)
    except Exception as error:
        # Keep chat flow alive even when Sheets auth/dependency fails.
        log(f'Erro ao carregar estoque no Sheets: {error}')
        return {}
    return grouped_products


def _flatten_products(categories_map: dict[str, dict]) -> dict[str, dict]:
    products: dict[str, dict] = {}
    for category in categories_map.values():
        for product in category.get('produtos', {}).values():
            code = str(product.get('codigo_pai', '')).strip().lower()
            if code and code not in products:
                products[code] = product
    return products


def _category_options(categories_map: dict[str, dict]) -> list[dict]:
    return [
        {
            'categoria': category.get('categoria', 'Sem categoria'),
            'produtos': category.get('produtos', {}),
        }
        for category in categories_map.values()
        if category.get('produtos')
    ]


def _fixed_category_options(categories_map: dict[str, dict]) -> list[dict]:
    normalized_categories = _category_options(categories_map)
    buckets = [
        {'categoria': 'Proteínas', 'produtos': {}},
        {'categoria': 'Aminoácidos', 'produtos': {}},
        {'categoria': 'Pré-treinos e Emagrecedores', 'produtos': {}},
        {'categoria': 'Vitaminas e Saúde', 'produtos': {}},
        {'categoria': 'Combos Promocionais', 'produtos': {}},
    ]

    keyword_groups = [
        {'proteina', 'proteínas', 'whey', 'albumina'},
        {'amino', 'aminoacido', 'aminoácido', 'creatina', 'bcaa', 'glutamina'},
        {'pre', 'pré', 'treino', 'termogenico', 'termogênico', 'emagrecedor'},
        {'vitamina', 'multivitaminico', 'multivitamínico', 'omega', 'ômega', 'saude', 'saúde'},
        {'combo', 'kit', 'promocao', 'promoção', 'coqueteleira', 'acessorio', 'acessório'},
    ]

    for category in normalized_categories:
        category_name = _normalize_text(category.get('categoria', ''))
        products = category.get('produtos', {})

        for product_key, product in products.items():
            product_name = _normalize_text(str(product.get('nome_produto', '')))
            variation_tokens = ' '.join(
                _normalize_text(str(variation.get('variacao', '')))
                for variation in product.get('variacoes', [])
            )
            product_context = f'{category_name} {product_name} {variation_tokens}'

            matched_bucket_index = None
            for index, keywords in enumerate(keyword_groups):
                if any(keyword in product_context for keyword in keywords):
                    matched_bucket_index = index
                    break

            if matched_bucket_index is None:
                matched_bucket_index = 4

            buckets[matched_bucket_index]['produtos'][product_key] = product

    return buckets


def _grouped_products_by_code(grouped_products: dict[str, dict]) -> dict[str, dict]:
    return _flatten_products(grouped_products)


def _catalog_for_intent_parser(products_by_code: dict[str, dict]) -> list[dict]:
    catalog: list[dict] = []
    for product in products_by_code.values():
        catalog.append(
            {
                'codigo_pai': str(product.get('codigo_pai', '')).strip(),
                'nome_produto': str(product.get('nome_produto', '')).strip(),
                'variacoes': [
                    str(variation.get('variacao', '')).strip() or 'Unico'
                    for variation in product.get('variacoes', [])
                ],
            }
        )
    return catalog


def _find_catalog_variation(product_group: dict, variation_name: str) -> dict | None:
    variations = product_group.get('variacoes', [])
    if not variations:
        return None

    normalized_variation = _normalize_text(variation_name or '')
    if not normalized_variation and len(variations) == 1:
        return variations[0]

    for variation in variations:
        current_name = str(variation.get('variacao', '')).strip() or 'Unico'
        if _normalize_text(current_name) == normalized_variation:
            return variation

    if normalized_variation in {'unico', 'único'} and len(variations) == 1:
        return variations[0]

    return None


def _build_cart_item_name(product_group: dict, variation: dict) -> str:
    variation_name = str(variation.get('variacao', '')).strip()
    base_name = str(product_group.get('nome_produto', '')).strip()

    if not variation_name or variation_name.lower() in {'unico', 'único'}:
        return base_name

    return f'{base_name} - {variation_name}'


def _add_item_to_cart(
    cart_items: list[dict],
    product_group: dict,
    variation: dict,
    quantity: int,
):
    parent_code = str(product_group.get('codigo_pai', '')).strip().lower()
    variation_name = str(variation.get('variacao', '')).strip() or 'Unico'
    variation_key = variation_name.lower()

    for item in cart_items:
        if item.get('code') == parent_code and item.get('variation', '').lower() == variation_key:
            item['quantity'] = int(item.get('quantity', 0)) + quantity
            return

    cart_items.append(
        {
            'code': parent_code,
            'variation': variation_name,
            'product_name': str(product_group.get('nome_produto', '')).strip(),
            'name': _build_cart_item_name(product_group, variation),
            'price': float(variation['preco']),
            'quantity': quantity,
        }
    )


async def _send_product_media_if_available(chat_id: str, product_group: dict, caption: str):
    media_url = str(product_group.get('imagem_url', '')).strip()
    if not _is_valid_media_url(media_url):
        return

    try:
        await asyncio.to_thread(
            send_whatsapp_media,
            chat_id,
            caption,
            media_url,
        )
    except Exception as error:
        log(f'Falha ao enviar imagem do produto: {error}')


async def _send_welcome_sequence(chat_id: str, instance_name: str):
    try:
        await asyncio.to_thread(
            send_whatsapp_message,
            chat_id,
            '💪 Bem-vindo(a) ao atendimento da Bora Treinar Suplementos! O seu parceiro na busca pelos melhores resultados. 🏆',
            instance_name,
        )
    except Exception as error:
        log(f'Falha ao enviar boas-vindas para {chat_id}: {error}')

    if not WELCOME_IMAGE_PATH.exists():
        log(f'Imagem de boas-vindas não encontrada em {WELCOME_IMAGE_PATH}')
        return

    try:
        await asyncio.to_thread(
            send_whatsapp_image_file,
            chat_id,
            str(WELCOME_IMAGE_PATH),
            '',
            instance_name,
        )
    except Exception as error:
        log(f'Falha ao enviar imagem de boas-vindas para {chat_id}: {error}')


def _build_admin_summary(pedido: dict) -> str:
    return (
        '🛎️ *Novo Pedido - Loja de Suplementos*\n\n'
        f"*Cliente:* {pedido['nome']}\n"
        f"*Telefone:* {pedido['numero']}\n"
        f"*Endereco:* {pedido['endereco']}\n"
        f"*Pagamento:* {pedido.get('forma_pagamento', 'Nao informado')}\n\n"
        f"*Itens:*\n{pedido['itens_resumo']}\n\n"
        f"*Total:* {format_brl(float(pedido['total']))}\n"
        f"*Status:* {pedido['status']}"
    )


async def _get_checkout_customer(chat_id: str, context: dict, tenant_id: str) -> tuple[str, str]:
    customer_name = context.get('customer_name', '').strip()
    customer_address = context.get('customer_address', '').strip()

    if customer_name and customer_address:
        return customer_name, customer_address

    try:
        existing_customer = await asyncio.to_thread(get_cliente_by_phone, chat_id, tenant_id)
    except Exception as error:
        log(f'Erro ao consultar cliente no checkout: {error}')
        return customer_name, customer_address

    if not existing_customer:
        return customer_name, customer_address

    return (
        customer_name or existing_customer.get('nome', '').strip(),
        customer_address or existing_customer.get('endereco', '').strip(),
    )


async def process_message(chat_id: str, user_message: str, tenant_id: str, instance_name: str) -> str:
    logger.info(f"[PROCESS] tenant_id={tenant_id} sub_nicho=<not_loaded_yet> msg='{user_message[:50]}'")
    # Modo IA pura: sem fluxos de menu/carrinho/checkout.
    # PARTE 1: ROTEADOR DE SCRIPT — detecta intenções e responde com script sem gastar tokens
    conversation_id = f'{tenant_id}:{chat_id}'

    try:
        tenant_configs = await get_tenant_configs(tenant_id)
        logger.info('[ROUTER] sub_nicho: %s', tenant_configs.get('sub_nicho'))

        produto_confirmado_key = f'session:{tenant_id}:{chat_id}:produto_confirmado'
        last_order_key = f'session:{tenant_id}:{chat_id}:last_order_time'
        last_order_raw = await redis_client.get(last_order_key)
        if last_order_raw:
            try:
                elapsed = time.time() - float(last_order_raw)
            except (TypeError, ValueError):
                elapsed = 0.0
            if elapsed > 300:
                await redis_client.delete(produto_confirmado_key)
                await clear_order_context(chat_id)
                clear_session_history(conversation_id)
                logger.info('[SESSION] Contexto limpo após %.0fs do último pedido', elapsed)
    except Exception as error:
        log(f'Erro ao carregar configs do tenant no banco: {error}')
        tenant_configs = {
            'promptIa': '',
            'whatsappAdmin': '',
            'botObjective': 'FECHAR_PEDIDO',
        }

    # 1.1 — Verificar se está fora do horário
    horarios = tenant_configs.get('horarios', [])
    if horarios and not is_within_hours(horarios):
        log(f'Chat {chat_id} fora do horário — retornando resposta de fechado')
        return resposta_fora_horario(tenant_configs)

    # Fluxo dedicado para pizzaria: evita loop de confirmação no fluxo geral de IA.
    sub_nicho = str(tenant_configs.get('sub_nicho') or '').strip().lower()
    if sub_nicho == 'pizzaria':
        session_key = f'pizza_session:{tenant_id}:{chat_id}'
        session_raw = await redis_client.get(session_key)
        try:
            session = json.loads(session_raw) if session_raw else {}
        except Exception:
            session = {}

        try:
            estoque = await fetch_stock_for_context(tenant_id)
        except Exception as error:
            log(f'Erro ao carregar cardápio de pizzaria: {error}')
            estoque = {}

        cardapio = estoque.get('sabores', []) if isinstance(estoque, dict) else []
        tamanhos = estoque.get('tamanhos', []) if isinstance(estoque, dict) else []
        bordas = estoque.get('bordas', []) if isinstance(estoque, dict) else []

        resposta, session_atualizada = process_pizza_message(
            text=user_message,
            session=session,
            cardapio=cardapio,
            tamanhos=tamanhos,
            bordas=bordas,
            tenant_config=tenant_configs,
        )

        await redis_client.setex(session_key, 1800, json.dumps(session_atualizada, ensure_ascii=False))

        if session_atualizada.get('state') == PizzaState.FINALIZADO.value:
            try:
                await save_pizza_order(tenant_id=tenant_id, phone=chat_id, session=session_atualizada)
            except Exception as error:
                log(f'Falha ao persistir pedido de pizzaria: {error}')
            await redis_client.delete(session_key)

        return resposta

    # Fluxo dedicado para adega
    if sub_nicho == 'adega':
        session_key = f'adega_session:{tenant_id}:{chat_id}'
        session_raw = await redis_client.get(session_key)
        try:
            session = json.loads(session_raw) if session_raw else {}
        except Exception:
            session = {}

        try:
            estoque_data = await fetch_stock_for_context(tenant_id)
        except Exception as error:
            log(f'Erro ao carregar estoque de adega: {error}')
            estoque_data = {}

        estoque = estoque_data.get('items', []) if isinstance(estoque_data, dict) else []

        resposta, session_atualizada = process_adega_message(
            text=user_message,
            session=session,
            estoque=estoque,
            tenant_config=tenant_configs,
        )

        if resposta is not None:
            await redis_client.setex(session_key, 1800, json.dumps(session_atualizada, ensure_ascii=False))
            if session_atualizada.get('state') == AdegaState.FINALIZADO.value:
                try:
                    payload = save_adega_order_payload(session_atualizada)
                    await save_order(
                        tenant_id=tenant_id,
                        phone=chat_id,
                        nome='Cliente WhatsApp',
                        endereco=payload['endereco'],
                        cart_items=[{
                            'product_name': payload['produto'],
                            'name': payload['produto'],
                            'quantity': payload['quantidade'],
                            'price': payload['total'] / max(payload['quantidade'], 1),
                        }],
                        total=payload['total'],
                        forma_pagamento=payload['pagamento'],
                    )
                except Exception as error:
                    log(f'Falha ao persistir pedido de adega: {error}')
                await redis_client.delete(session_key)
            return resposta
        # resposta é None → continuar para intent detection e fallback de IA

    # Fluxo dedicado para lanchonete
    if sub_nicho == 'lanchonete':
        session_key = f'lanchonete_session:{tenant_id}:{chat_id}'
        session_raw = await redis_client.get(session_key)
        try:
            session = json.loads(session_raw) if session_raw else {}
        except Exception:
            session = {}

        try:
            estoque_data = await fetch_stock_for_context(tenant_id)
        except Exception as error:
            log(f'Erro ao carregar estoque de lanchonete: {error}')
            estoque_data = {}

        estoque = estoque_data.get('items', []) if isinstance(estoque_data, dict) else []

        resposta, session_atualizada = process_lanchonete_message(
            text=user_message,
            session=session,
            estoque=estoque,
            tenant_config=tenant_configs,
        )

        if resposta is not None:
            await redis_client.setex(session_key, 1800, json.dumps(session_atualizada, ensure_ascii=False))
            if session_atualizada.get('state') == LanchoneteState.FINALIZADO.value:
                try:
                    payload = save_lanchonete_order_payload(session_atualizada)
                    cart_items = [
                        {
                            'product_name': item['nome'],
                            'name': item['nome'],
                            'quantity': item['quantidade'],
                            'price': item['preco'],
                        }
                        for item in payload['carrinho']
                    ]
                    await save_order(
                        tenant_id=tenant_id,
                        phone=chat_id,
                        nome='Cliente WhatsApp',
                        endereco=payload['endereco'],
                        cart_items=cart_items,
                        total=payload['total'],
                        forma_pagamento=payload['pagamento'],
                    )
                except Exception as error:
                    log(f'Falha ao persistir pedido de lanchonete: {error}')
                await redis_client.delete(session_key)
            return resposta
        # resposta é None → continuar para intent detection e fallback de IA

    # 1.2 — Detectar intenção de script
    intent = detect_intent(user_message)
    log(f'Intenção detectada para {chat_id}: {intent}')

    # 1.3 — Responder com script baseado em intenção
    if intent:
        try:
            if intent == 'saudacao':
                # Saudações: responder só se for primeira mensagem
                historico = get_session_history(conversation_id)
                if not historico.messages:
                    log(f'Chat {chat_id} — cumprimento em sessão nova')
                    return resposta_saudacao(tenant_configs)
                # Se já tem histórico, deixar IA responder naturalmente

            elif intent == 'horario':
                log(f'Chat {chat_id} — pergunta sobre horário')
                return resposta_horario(tenant_configs)

            elif intent == 'entrega':
                log(f'Chat {chat_id} — pergunta sobre entrega')
                return resposta_entrega(tenant_configs)

            elif intent == 'status_pedido':
                log(f'Chat {chat_id} — pergunta sobre status')
                ultimo_pedido = await asyncio.to_thread(
                    get_ultimo_pedido,
                    phone=chat_id,
                    tenant_id=tenant_id,
                )
                return resposta_status_pedido(ultimo_pedido)

            elif intent == 'cardapio':
                log(f'Chat {chat_id} — pedido de cardápio')
                estoque = await fetch_stock_for_context(tenant_id)
                return resposta_cardapio(tenant_configs, estoque)

            elif intent == 'cancelar':
                log(f'Chat {chat_id} — solicitação cancelamento')
                return resposta_cancelamento_confirmacao(tenant_configs)

            elif intent == 'fechar_pedido':
                log(f'Chat {chat_id} — intenção de fechamento de pedido')
                history = get_session_history(conversation_id)
                payload = await build_order_payload_from_history_window_async(
                    history_window=list(history.messages),
                    user_message=user_message,
                    tenant_id=tenant_id,
                )
                order_data = payload.get('order') or {}
                items = order_data.get('items') or []
                address = str(order_data.get('customer_address') or '').strip()
                payment = str(order_data.get('payment_method') or '').strip()

                if items and not address:
                    if payment:
                        return f'Anotado! Pagamento no {payment}. Me manda o endereço pra entrega 📍'
                    return 'Ótimo! Me manda o endereço pra entrega 📍'

                if items and address and not payment:
                    return 'Qual forma de pagamento? Pix, dinheiro ou cartão?'

                if items and address and payment:
                    cart_items = [
                        {
                            'product_name': str(item.get('product_name') or '').strip(),
                            'name': str(item.get('product_name') or '').strip(),
                            'quantity': max(1, int(item.get('quantity') or 1)),
                            'price': float(item.get('price') or 0),
                        }
                        for item in items
                        if str(item.get('product_name') or '').strip()
                    ]
                    summary, _ = build_checkout_summary(
                        cart_items=cart_items,
                        address=address,
                        payment_method=payment,
                        nome_negocio=str(tenant_configs.get('nome_negocio') or 'nossa loja'),
                        tenant_id=tenant_id,
                    )
                    if summary:
                        return summary

                return 'Perfeito! Me confirma só o item e a quantidade para eu fechar certinho.'

        except Exception as error:
            log(f'Erro ao processar intenção {intent}: {error}')
            # Cai no fluxo normal da IA se houver erro

    # PARTE 2: FLUXO IA NORMAL — nenhum script bateu, chamar Gemini
    log(f'Chat {chat_id} — roteador não bateu, enviando para IA')
    prompt_ia = tenant_configs.get('promptIa') or None
    bot_objective = tenant_configs.get('botObjective') or 'FECHAR_PEDIDO'
    instruction = (
        'Responda ao cliente de forma natural, útil e objetiva. '
        'Não use menus numerados nem conduza fluxo fixo. '
        'Se faltar contexto, faça uma pergunta curta para clarificar.'
    )

    try:
        return await asyncio.to_thread(
            generate_persona_response,
            instruction,
            user_message,
            conversation_id,
            prompt_ia,
            bot_objective,
            tenant_id,
        )
    except Exception as error:
        log(f'Erro ao gerar resposta da IA: {error}')
        return (
            '⚠️ Estou com instabilidade momentânea para responder agora. '
            'Pode repetir sua mensagem em alguns segundos?'
        )


async def buffer_message(chat_id: str, message: str, tenant_id: str, instance_name: str):
    session_key = f'{tenant_id}:{chat_id}'
    buffer_key = f'{session_key}{BUFFER_KEY_SUFIX}'

    await redis_client.rpush(buffer_key, message)
    await redis_client.expire(buffer_key, BUFFER_TTL)

    log(f'Mensagem adicionada ao buffer de {session_key}: {message}')

    if debounce_tasks.get(session_key):
        debounce_tasks[session_key].cancel()
        log(f'Debounce resetado para {session_key}')

    debounce_tasks[session_key] = asyncio.create_task(
        handle_debounce(chat_id, tenant_id, instance_name)
    )


async def handle_debounce(chat_id: str, tenant_id: str, instance_name: str):
    session_key = f'{tenant_id}:{chat_id}'
    buffer_key = f'{session_key}{BUFFER_KEY_SUFIX}'

    try:
        log(f'Iniciando debounce para {session_key}')
        await asyncio.sleep(float(DEBOUNCE_SECONDS))

        messages = await redis_client.lrange(buffer_key, 0, -1)

        full_message = ' '.join(messages).strip()
        if full_message:
            log(f'Enviando mensagem agrupada para {chat_id}: {full_message}')
            reply_message = await process_message(chat_id, full_message, tenant_id, instance_name)

            if not reply_message:
                await redis_client.delete(buffer_key)
                return

            try:
                send_whatsapp_presence(
                    number=chat_id,
                    presence='composing',
                    delay=300,
                    instance_name=instance_name,
                )
            except Exception as error:
                log(f'Falha ao enviar presença para {chat_id}: {error}')

            try:
                send_whatsapp_message(
                    number=chat_id,
                    text=reply_message,
                    instance_name=instance_name,
                )
                log(f'Resposta enviada para {chat_id}')
            except Exception as error:
                log(f'Falha ao enviar mensagem para {chat_id}: {error}')

            # Persistência de pedido não deve depender do envio no WhatsApp.
            # Se o resumo final foi gerado, salva no Kanban mesmo com falha de transporte.
            try:
                await _persist_ai_final_order_if_needed(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    reply_message=reply_message,
                )
            except Exception as error:
                log(f'Falha ao persistir pedido após debounce para {chat_id}: {error}')
        await redis_client.delete(buffer_key)

    except asyncio.CancelledError:
        log(f'Debounce cancelado para {session_key}')
    except Exception as error:
        log(f'Erro inesperado no debounce de {session_key}: {error}')
        fallback_message = (
            '⚠️ Tive um problema técnico para processar sua mensagem agora. '
            'Tente novamente em instantes ou digite *menu* para continuar.'
        )
        if 'invalid_api_key' in str(error).lower() or 'incorrect api key' in str(error).lower():
            fallback_message = (
                '⚠️ Nosso assistente está temporariamente indisponível por configuração interna. '
                'Digite *menu* para continuar no atendimento ou tente novamente em instantes.'
            )
        try:
            send_whatsapp_message(number=chat_id, text=fallback_message, instance_name=instance_name)
        except Exception as send_error:
            log(f'Falha ao enviar fallback para {chat_id}: {send_error}')
        try:
            await redis_client.delete(buffer_key)
        except Exception as delete_error:
            log(f'Falha ao limpar buffer de {chat_id}: {delete_error}')
    finally:
        current_task = debounce_tasks.get(session_key)
        running_task = asyncio.current_task()
        if current_task is running_task:
            debounce_tasks.pop(session_key, None)
