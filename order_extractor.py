from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from typing import Any
import redis.asyncio as redis
from difflib import SequenceMatcher

try:
    from rapidfuzz.fuzz import token_set_ratio as _rf_token_set_ratio  # pyright: ignore[reportMissingImports]
except Exception:
    def _rf_token_set_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100

from database_api import fetch_stock_for_context
from memory import get_session_history
from config import REDIS_URL

logger = logging.getLogger(__name__)


PLACEHOLDER_ITEMS = '[itens informados pelo cliente]'
PLACEHOLDER_ADDRESS = '[endereço informado pelo cliente]'
PLACEHOLDER_PAYMENT = '[forma escolhida]'

_PRODUCT_CONFIRMED_TTL_SECONDS = 30 * 60
_redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None


def _produto_confirmado_key(tenant_id: str, phone: str) -> str:
    return f'session:{tenant_id}:{phone}:produto_confirmado'

_TOKEN_STOPWORDS = {
    'a', 'o', 'as', 'os', 'de', 'do', 'da', 'dos', 'das', 'e', 'ou', 'uma', 'um',
    'pedido', 'quero', 'pizza', 'pizzas', 'sabor', 'sabores', 'no', 'na', 'com',
}

_TOKEN_CANONICAL = {
    'c': 'com',
    'catupiri': 'catupiry',
    'mucarela': 'mussarela',
    'muzzarela': 'mussarela',
}

_CANDIDATE_NOISE_TOKENS = {
    'tem', 'como', 'me', 'mandar', 'manda', 'pode', 'ser', 'ai', 'aqui', 'pra', 'para',
}


def is_final_order_message_text(text: str) -> bool:
    normalized = (text or '').strip()
    return '*Resumo do Pedido*' in normalized and normalized.endswith('✅')


def slice_messages_after_last_completed_order(messages: list[Any]) -> list[Any]:
    last_final_index = -1
    for index, message in enumerate(messages):
        if getattr(message, 'type', '') != 'ai':
            continue
        if is_final_order_message_text(str(getattr(message, 'content', '') or '')):
            last_final_index = index

    if last_final_index < 0:
        return list(messages)

    return list(messages[last_final_index + 1:])


def collect_active_human_texts(messages: list[Any], include_user_message: str = '') -> list[str]:
    active_messages = slice_messages_after_last_completed_order(messages)
    human_texts = [
        str(message.content).strip()
        for message in active_messages
        if getattr(message, 'type', '') == 'human' and str(message.content).strip()
    ]
    if include_user_message.strip():
        human_texts.append(include_user_message.strip())
    return human_texts


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize('NFKD', value or '')
    normalized = ''.join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r'[^a-z0-9]+', ' ', normalized)
    return ' '.join(normalized.split())


def _normalized_tokens(value: str) -> list[str]:
    tokens = normalize_text(value).split()
    canonical_tokens: list[str] = []
    for token in tokens:
        canonical = _TOKEN_CANONICAL.get(token, token)
        if canonical in _TOKEN_STOPWORDS:
            continue
        canonical_tokens.append(canonical)
    return canonical_tokens


def _token_match_score(source_text: str, candidate_name: str) -> float:
    source_tokens = set(_normalized_tokens(source_text))
    candidate_tokens = set(_normalized_tokens(candidate_name))
    if not source_tokens or not candidate_tokens:
        return 0.0

    intersection = source_tokens.intersection(candidate_tokens)
    if not intersection:
        compact_source = re.sub(r'[^a-z0-9]+', '', normalize_text(source_text))
        compact_candidate = re.sub(r'[^a-z0-9]+', '', normalize_text(candidate_name))
        compact_candidate_base = re.sub(r'\d+', '', compact_candidate)
        compact_candidate_base = re.sub(r'(ml|kg|g|l)+$', '', compact_candidate_base)
        if compact_candidate and compact_candidate in compact_source:
            return 1.0
        if compact_candidate_base and compact_candidate_base in compact_source:
            return 1.0
        return 0.0

    # Score baseado em cobertura do nome do produto.
    return len(intersection) / len(candidate_tokens)


def contains_order_signal(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        token in normalized
        for token in (
            'quero',
            'pedido',
            'pizza',
            'lanche',
            'hamburg',
            'combo',
            'calabresa',
            'frango',
            'portuguesa',
            'mussarela',
            'muzzarela',
        )
    )


def extract_payment_from_text(text: str) -> str:
    normalized = normalize_text(text)
    if 'pix' in normalized:
        return 'Pix'
    if any(token in normalized for token in ('cartao', 'credito', 'debito')):
        return 'Cartão'
    if 'dinheiro' in normalized or 'dinheirinho' in normalized:
        return 'Dinheiro'
    return ''


def extract_name_from_text(text: str) -> str:
    patterns = (
        r'(?i)meu nome e\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
        r'(?i)pode colocar no nome de\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
        r'(?i)nome[:\-]?\s*([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
        r'(?i)sou o\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
        r'(?i)sou a\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})',
    )

    for pattern in patterns:
        match = re.search(pattern, text or '')
        if match:
            return (match.group(1) or '').strip()

    return ''


def extract_address_from_text(text: str) -> str:
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


def address_looks_broken(value: str) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return True

    return any(
        token in normalized
        for token in ('tamanho', 'pagamento', 'pix', 'cartao', 'dinheiro')
    )


def _is_affirmative_confirmation(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    return any(
        phrase in normalized
        for phrase in (
            'sim',
            'isso',
            'exato',
            'esse',
            'sim esse',
            'sim esse mesmo',
            'isso mesmo',
            'pode ser',
            'fechou',
            'confirmo',
        )
    )


def extract_requested_size_label(text: str) -> str:
    normalized = normalize_text(text)
    patterns = (
        (r'\bgg\b|gigante|familia', 'GG'),
        (r'\bgrande\b|\bg\b', 'G'),
        (r'\bmedia\b|\bmedio\b|\bm\b', 'M'),
        (r'\bpequena\b|\bpequeno\b|\bp\b', 'P'),
    )

    for pattern, label in patterns:
        if re.search(pattern, normalized):
            return label

    return ''


def _collect_candidate_names(stock_data: dict[str, Any]) -> list[str]:
    sub_nicho = str(stock_data.get('sub_nicho') or '').strip().lower()
    if sub_nicho == 'pizzaria':
        return [
            str(sabor.get('nome') or '').strip()
            for sabor in stock_data.get('sabores', [])
            if str(sabor.get('nome') or '').strip()
        ]

    return [
        str(item.get('nome') or '').strip()
        for item in stock_data.get('items', [])
        if str(item.get('nome') or '').strip()
    ]


def _best_catalog_candidate(product_source: str, stock_data: dict[str, Any]) -> dict[str, Any] | None:
    best_name = ''
    best_score = 0.0
    normalized_source = normalize_text(product_source)
    if not normalized_source:
        return None

    for candidate in _collect_candidate_names(stock_data):
        normalized_candidate = normalize_text(candidate)
        if not normalized_candidate:
            continue

        score = _token_match_score(product_source, candidate)
        if normalized_candidate in normalized_source:
            score = max(score, 1.0)

        if score > best_score:
            best_name = candidate
            best_score = score

    if not best_name:
        return None

    return {
        'product_name': best_name,
        'score': best_score,
    }


def _strip_leading_order_prefix(text: str) -> str:
    tokens = text.split()
    prefixes = {
        'quero', 'queria', 'traz', 'manda', 'pedido', 'pedir',
        'me', 'da', 'dar', 'pode', 'ser', 'vou', 'querer',
    }
    while tokens and tokens[0] in prefixes:
        tokens.pop(0)
    return ' '.join(tokens).strip()


def extract_quantity_product_pairs(text: str) -> list[tuple[int, str]]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    word_to_num = {
        'um': 1,
        'uma': 1,
        'dois': 2,
        'duas': 2,
        'tres': 3,
        'quatro': 4,
        'cinco': 5,
        'seis': 6,
        'sete': 7,
        'oito': 8,
        'nove': 9,
        'dez': 10,
    }

    text_norm = normalized
    for word, number in word_to_num.items():
        text_norm = re.sub(rf'\b{word}\b', str(number), text_norm)

    # Normaliza "2x heineken" e "2 x heineken" para facilitar parsing por tokens.
    text_norm = re.sub(r'\b(\d+)x\b', r'\1', text_norm)
    text_norm = re.sub(r'\b(\d+)\s+x\s+', r'\1 ', text_norm)

    heuristic_pairs: list[tuple[int, str]] = []

    # Ex.: "tem redbull? quero 20" -> (20, redbull)
    for match in re.finditer(r'\btem\s+([^\?\.,]+?)\s*[\?\.,]?\s*quero\s+(\d+)\b', text_norm):
        raw_product = (match.group(1) or '').strip()
        qty = max(1, int(match.group(2)))
        product_tokens = [tok for tok in raw_product.split() if tok and tok not in _CANDIDATE_NOISE_TOKENS]
        product = ' '.join(product_tokens).strip()
        if product:
            heuristic_pairs.append((qty, product))

    # Ex.: "quero 20 red bull" -> (20, red bull)
    for match in re.finditer(r'\bquero\s+(\d+)\s+(?:de\s+)?([^\?\.,]+)', text_norm):
        qty = max(1, int(match.group(1)))
        raw_product = (match.group(2) or '').strip()
        raw_product = re.split(r'\b(tem|me manda|manda|por favor|pfv|pfv\.)\b', raw_product)[0].strip()
        product_tokens = [tok for tok in raw_product.split() if tok and tok not in _CANDIDATE_NOISE_TOKENS]
        product = ' '.join(product_tokens).strip()
        if product:
            heuristic_pairs.append((qty, product))

    if heuristic_pairs:
        return heuristic_pairs

    tokens = text_norm.split()
    pairs: list[tuple[int, str]] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.isdigit():
            idx += 1
            continue

        quantity = max(1, int(token))
        idx += 1

        product_tokens: list[str] = []
        while idx < len(tokens):
            current = tokens[idx]
            if current.isdigit():
                break
            if current == 'e' and idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
                break
            product_tokens.append(current)
            idx += 1

        while product_tokens and product_tokens[-1] in {'e', 'de', 'da', 'do', 'para', 'pra'}:
            product_tokens.pop()

        product_raw = ' '.join(product_tokens).strip()
        if product_raw:
            pairs.append((quantity, product_raw))

    if pairs:
        return pairs

    fallback = _strip_leading_order_prefix(text_norm)
    if fallback:
        fallback_pairs = [(1, fallback)]
        return fallback_pairs

    return []


def extract_quantity_from_confirmation(text: str) -> int:
    """Extrai quantidade de mensagens de confirmação como "sim, quero 5"."""
    normalized = normalize_text(text)
    patterns = [
        r'(?:sim|pode|certo|isso)[,\s]+(?:quero\s+)?(\d+)',
        r'(?:quero|me\s+manda|me\s+da)\s+(\d+)',
        r'^(\d+)$',
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                qty = int(match.group(1))
                return qty if qty > 0 else 1
            except (TypeError, ValueError):
                return 1

    return 1


def extract_quantity_from_history(history: list[str]) -> int:
    """Retorna a última quantidade explícita encontrada no histórico (mais recente primeiro)."""
    for message in reversed(history):
        qty = extract_quantity_from_confirmation(message)
        if qty > 1:
            return qty
    return 1


def extract_confirmed_product_from_history(
    history: list[str],
    catalog: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Busca no histórico recente o último produto confirmado por similaridade."""
    for message in reversed(history):
        normalized_message = normalize_text(message)
        if not normalized_message:
            continue

        best_item: dict[str, Any] | None = None
        best_score = 0.0
        for item in catalog:
            item_name = str(item.get('nome') or '').strip()
            if not item_name:
                continue

            score = _rf_token_set_ratio(normalized_message, normalize_text(item_name)) / 100
            if score > best_score:
                best_score = score
                best_item = item

        if best_item and best_score >= 0.6:
            return best_item

    return None


def _catalog_items_from_stock(stock_data: dict[str, Any]) -> list[dict[str, Any]]:
    sub_nicho = str(stock_data.get('sub_nicho') or '').strip().lower()
    if sub_nicho == 'pizzaria':
        return [
            {'nome': str(item.get('nome') or '').strip()}
            for item in stock_data.get('sabores', [])
            if str(item.get('nome') or '').strip()
        ]

    return [
        {'nome': str(item.get('nome') or '').strip()}
        for item in stock_data.get('items', [])
        if str(item.get('nome') or '').strip()
    ]


def extract_catalog_items_from_text(product_source: str, stock_data: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_source = normalize_text(product_source)
    if not normalized_source:
        return []

    candidates = _collect_candidate_names(stock_data)

    explicit_best = _best_catalog_candidate(product_source, stock_data)

    pairs = extract_quantity_product_pairs(product_source)
    if not pairs:
        return []

    best_matches: list[dict[str, Any]] = []
    for quantity, product_raw in pairs:
        best_name = ''
        best_score = 0.0
        normalized_raw = normalize_text(product_raw)
        for candidate in candidates:
            normalized_candidate = normalize_text(candidate)
            if not normalized_candidate:
                continue
            score = _token_match_score(product_raw, candidate)
            if normalized_candidate in normalized_raw:
                score = max(score, 1.0)
            if score > best_score:
                best_name = candidate
                best_score = score

        if best_name and best_score >= 0.45:
            best_matches.append(
                {
                    'product_name': best_name,
                    'quantity': quantity,
                    'score': best_score,
                }
            )

    if not best_matches:
        qty_from_text = extract_quantity_from_confirmation(product_source)
        if explicit_best and float(explicit_best.get('score') or 0) >= 0.45:
            explicit_name = str(explicit_best.get('product_name') or '').strip()
            if explicit_name:
                return [
                    {
                        'product_name': explicit_name,
                        'base_product_name': explicit_name,
                        'size': '',
                        'quantity': max(1, qty_from_text),
                        'price': 0.0,
                        'match_source': 'catalog_explicit',
                    }
                ]
        return []

    size_label = extract_requested_size_label(product_source)
    extracted_items: list[dict[str, Any]] = []
    for match in best_matches:
        product_name = str(match.get('product_name') or '').strip()
        if not product_name:
            continue
        quantity = max(1, int(match.get('quantity') or 1))
        display_name = f'{product_name} ({size_label})' if size_label else product_name
        extracted_items.append(
            {
                'product_name': display_name,
                'base_product_name': product_name,
                'size': size_label,
                'quantity': quantity,
                'price': 0.0,
                'match_source': 'catalog',
            }
        )

    return extracted_items


def build_items_text(items: list[dict[str, Any]]) -> str:
    if not items:
        return PLACEHOLDER_ITEMS

    return ' + '.join(
        f"{max(1, int(item.get('quantity') or 1))}x {str(item.get('product_name') or '').strip()}"
        for item in items
        if str(item.get('product_name') or '').strip()
    ) or PLACEHOLDER_ITEMS


def _fallback_items_from_text(items_text: str) -> list[dict[str, Any]]:
    normalized_items = (items_text or '').strip()
    if not normalized_items:
        return []

    parts = [part.strip() for part in normalized_items.split('+') if part.strip()]
    parsed_items: list[dict[str, Any]] = []
    for part in parts:
        quantity_match = re.match(r'^\s*(\d+)\s*[xX]\s+(.+)$', part)
        if quantity_match:
            quantity = int(quantity_match.group(1))
            product_name = quantity_match.group(2).strip()
        else:
            quantity = 1
            product_name = part

        if not product_name:
            continue

        parsed_items.append(
            {
                'product_name': product_name,
                'base_product_name': product_name,
                'size': '',
                'quantity': max(1, quantity),
                'price': 0.0,
                'match_source': 'summary_fallback',
            }
        )

    return parsed_items


def build_order_payload_from_texts(
    human_texts: list[str],
    tenant_id: str | None,
    stock_data: dict[str, Any],
    fallback_items_text: str = '',
    fallback_address: str = '',
    fallback_payment: str = '',
) -> dict[str, Any]:
    produto_confirmado = False
    order_messages = [text for text in human_texts if contains_order_signal(text)]
    candidate_sources = list(reversed(order_messages))
    if not candidate_sources and human_texts:
        candidate_sources = [human_texts[-1]]

    product_source = candidate_sources[0] if candidate_sources else ''

    customer_name = ''
    customer_address = '' if address_looks_broken(fallback_address) else fallback_address.strip()
    payment_method = fallback_payment.strip()

    for text in reversed(human_texts):
        if not customer_name:
            customer_name = extract_name_from_text(text)
        if not customer_address:
            customer_address = extract_address_from_text(text)
        if not payment_method:
            payment_method = extract_payment_from_text(text)
        if customer_name and customer_address and payment_method:
            break

    items: list[dict[str, Any]] = []
    suggestion_name = ''
    suggestion_score = 0.0
    for source in candidate_sources[:5]:
        current_items = extract_catalog_items_from_text(source, stock_data)
        if current_items:
            items = current_items
            product_source = source
            break

        suggestion = _best_catalog_candidate(source, stock_data)
        if suggestion and float(suggestion.get('score') or 0) > suggestion_score:
            suggestion_name = str(suggestion.get('product_name') or '').strip()
            suggestion_score = float(suggestion.get('score') or 0)

    if not items and order_messages:
        merged_source = ' '.join(order_messages[-3:])
        merged_items = extract_catalog_items_from_text(merged_source, stock_data)
        if merged_items:
            items = merged_items
            product_source = merged_source
        else:
            suggestion = _best_catalog_candidate(merged_source, stock_data)
            if suggestion and float(suggestion.get('score') or 0) > suggestion_score:
                suggestion_name = str(suggestion.get('product_name') or '').strip()
                suggestion_score = float(suggestion.get('score') or 0)

    if not items and fallback_items_text:
        items = _fallback_items_from_text(fallback_items_text)

    explicit_product_detected = False
    if not items and candidate_sources:
        for source in candidate_sources[:3]:
            suggestion = _best_catalog_candidate(source, stock_data)
            if suggestion and float(suggestion.get('score') or 0) >= 0.45:
                explicit_product_detected = True
                break

    if not items and human_texts:
        last_message = human_texts[-1]
        current_message_items = extract_catalog_items_from_text(last_message, stock_data)
        if current_message_items:
            items = current_message_items
            product_source = last_message
            logger.info(f"[EXTRATOR] Produto novo detectado na mensagem: {[(i.get('product_name'), i.get('quantity')) for i in current_message_items]}")

        qty = extract_quantity_from_confirmation(last_message)
        if qty <= 1:
            qty = extract_quantity_from_history(human_texts[:-1])
        if not items and not explicit_product_detected:
            confirmed = extract_confirmed_product_from_history(
                history=human_texts[:-1],
                catalog=_catalog_items_from_stock(stock_data),
            )
            if confirmed:
                confirmed_name = str(confirmed.get('nome') or '').strip()
                if confirmed_name:
                    size_label = extract_requested_size_label(' '.join(order_messages[-3:]))
                    display_name = f'{confirmed_name} ({size_label})' if size_label else confirmed_name
                    items = [
                        {
                            'product_name': display_name,
                            'base_product_name': confirmed_name,
                            'size': size_label,
                            'quantity': max(1, qty),
                            'price': 0.0,
                            'match_source': 'history_confirmation',
                        }
                    ]
                    logger.info(f"[EXTRATOR] Produto do histórico: {confirmed_name} x{max(1, qty)}")

    if not items and suggestion_name and suggestion_score >= 0.45 and human_texts:
        last_message = human_texts[-1]
        if _is_affirmative_confirmation(last_message):
            produto_confirmado = True
            size_label = extract_requested_size_label(' '.join(order_messages[-3:]))
            display_name = f'{suggestion_name} ({size_label})' if size_label else suggestion_name
            items = [
                {
                    'product_name': display_name,
                    'base_product_name': suggestion_name,
                    'size': size_label,
                    'quantity': 1,
                    'price': 0.0,
                    'match_source': 'suggestion_confirmation',
                }
            ]

    items_text = build_items_text(items)
    source_kind = 'single_message' if len(order_messages) <= 1 else 'multi_turn'
    captured_fields = [
        field_name
        for field_name, value in (
            ('customer_name', customer_name),
            ('customer_address', customer_address),
            ('payment_method', payment_method),
            ('items', items),
        )
        if value
    ]

    return {
        'version': 'order-extractor-v1',
        'tenant_id': (tenant_id or '').strip(),
        'order': {
            'items': items,
            'items_text': items_text,
            'customer_name': customer_name or 'Cliente WhatsApp',
            'customer_address': customer_address or '',
            'payment_method': payment_method or '',
            'sub_nicho': str(stock_data.get('sub_nicho') or '').strip().lower(),
            'suggested_product_name': suggestion_name,
            'suggested_product_score': suggestion_score,
        },
        'analytics': {
            'source_kind': source_kind,
            'order_messages_count': len(order_messages),
            'raw_last_order_message': product_source,
            'matched_product_names': [str(item.get('base_product_name') or '') for item in items],
            'captured_fields': captured_fields,
            'suggested_product_name': suggestion_name,
            'suggested_product_score': suggestion_score,
            'ready_for_premium_insights': True,
            'produto_confirmado': produto_confirmado,
        },
    }


def build_order_payload_from_history_window(
    history_window,
    user_message: str,
    tenant_id: str | None,
) -> dict[str, Any]:
    human_texts = collect_active_human_texts(list(history_window), include_user_message=user_message)

    stock_data: dict[str, Any] = {}
    normalized_tenant_id = (tenant_id or '').strip()
    if normalized_tenant_id:
        try:
            stock_data = asyncio.run(fetch_stock_for_context(normalized_tenant_id))
        except Exception:
            stock_data = {}

    return build_order_payload_from_texts(
        human_texts=human_texts,
        tenant_id=normalized_tenant_id,
        stock_data=stock_data,
    )


async def build_order_payload_from_history_window_async(
    history_window,
    user_message: str,
    tenant_id: str | None,
) -> dict[str, Any]:
    human_texts = collect_active_human_texts(list(history_window), include_user_message=user_message)

    stock_data: dict[str, Any] = {}
    normalized_tenant_id = (tenant_id or '').strip()
    if normalized_tenant_id:
        try:
            stock_data = await fetch_stock_for_context(normalized_tenant_id)
        except Exception:
            stock_data = {}

    return build_order_payload_from_texts(
        human_texts=human_texts,
        tenant_id=normalized_tenant_id,
        stock_data=stock_data,
    )