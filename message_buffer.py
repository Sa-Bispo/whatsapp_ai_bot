import asyncio
import importlib
import json
from pathlib import Path
import redis.asyncio as redis

from collections import defaultdict

from config import (
    REDIS_URL,
    BUFFER_KEY_SUFIX,
    DEBOUNCE_SECONDS,
    BUFFER_TTL,
    ADMIN_WHATSAPP_NUMBER,
)
from evolution_api import send_whatsapp_message, send_whatsapp_presence
try:
    from evolution_api import send_whatsapp_image_file, send_whatsapp_media
except ImportError:
    def send_whatsapp_image_file(number, file_path, caption='', instance_name=None):
        return None

    def send_whatsapp_media(number, caption, media_url, mediatype='image', file_name=None, mimetype=None, instance_name=None):
        return None
from database_api import (
    get_cliente_by_phone,
    get_ultimo_pedido,
    list_estoque,
    save_order,
)
from chains import generate_persona_response, invoke_rag_chain
try:
    _gemini_parser = importlib.import_module('gemini_parser')
    analyze_message = _gemini_parser.analyze_message
    extract_order_intent = _gemini_parser.extract_order_intent
except ImportError:
    async def analyze_message(user_message: str, cart_items: list[dict], parser_catalog: list[dict]) -> dict:
        normalized = (user_message or '').strip().lower()

        if any(token in normalized for token in {'atendente', 'humano', 'pessoa real', 'falar com alguem', 'falar com alguém'}):
            return {'intencao': 'atendimento_humano'}

        if any(token in normalized for token in {'carrinho', 'ver pedido', 'meu pedido'}):
            return {'intencao': 'ver_carrinho'}

        if any(token in normalized for token in {'finalizar', 'fechar pedido', 'concluir pedido', 'checkout'}):
            return {'intencao': 'checkout'}

        if any(token in normalized for token in {'duvida', 'dúvida', 'como', 'diferença', 'serve', 'funciona'}) and any(
            token in normalized for token in {'whey', 'creatina', 'bcaa', 'glutamina', 'pre treino', 'pré treino'}
        ):
            return {'intencao': 'duvida_tecnica'}

        tokens = normalized.replace(',', ' ').split()
        qty = 1
        for token in tokens:
            if token.isdigit():
                qty = max(1, int(token))
                break

        for product in parser_catalog or []:
            code = str(product.get('codigo_pai') or '').strip().lower()
            name = str(product.get('nome_produto') or '').strip().lower()
            variations = product.get('variacoes') or []
            variation = str(variations[0] if variations else 'Unico')

            code_match = code and any(token == code for token in tokens)
            name_match = name and name in normalized
            if not code_match and not name_match:
                continue

            return {
                'intencao': 'adicionar_carrinho',
                'status_item': 'completo',
                'dados_faltantes': [],
                'upsell_sugerido': None,
                'produto_identificado': {
                    'codigo_pai': code,
                    'variacao': variation,
                    'quantidade': qty,
                },
            }

        return {'intencao': ''}

    def extract_order_intent(*args, **kwargs):
        return {}


redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
debounce_tasks = defaultdict(asyncio.Task)

FLOW_STATE_SUFFIX = '_flow_state'
FLOW_CONTEXT_SUFFIX = '_flow_context'
FLOW_TENANT_SUFFIX = '_flow_tenant'
FLOW_INSTANCE_SUFFIX = '_flow_instance'
FLOW_CONTEXT_TTL = 60 * 60 * 24

STATE_VERIFICACAO_INICIAL = 'VERIFICACAO_INICIAL'
STATE_MENU_INICIAL = 'MENU_INICIAL'
STATE_DUVIDAS_SUPLEMENTOS = 'DUVIDAS_SUPLEMENTOS'
STATE_CATALOGO = 'CATALOGO'
STATE_ESCOLHENDO_CATEGORIA = 'ESCOLHENDO_CATEGORIA'
STATE_ADICIONANDO_CARRINHO = 'ADICIONANDO_CARRINHO'
STATE_AGUARDANDO_VARIACAO = 'AGUARDANDO_VARIACAO'
STATE_SUGERINDO_UPSELL = 'SUGERINDO_UPSELL'
STATE_CHECKOUT_NOME = 'CHECKOUT_NOME'
STATE_CHECKOUT_ENDERECO = 'CHECKOUT_ENDERECO'
STATE_CHECKOUT_PAGAMENTO = 'CHECKOUT_PAGAMENTO'
STATE_FINALIZACAO_CONFIRMACAO = 'FINALIZACAO_CONFIRMACAO'
STATE_ATENDIMENTO_HUMANO = 'ATENDIMENTO_HUMANO'

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


def _tenant_key(chat_id: str) -> str:
    return f'{chat_id}{FLOW_TENANT_SUFFIX}'


def _instance_key(chat_id: str) -> str:
    return f'{chat_id}{FLOW_INSTANCE_SUFFIX}'


def get_main_menu_text() -> str:
    return (
        'Como posso te ajudar a focar no treino hoje? (Digite o número da opção desejada):\n\n'
        '1️⃣ *Fazer um pedido / Ver Catálogo* 🛒\n'
        '2️⃣ *Dúvidas sobre suplementos (Whey, Creatina, Pré-treino...)* 🤔\n'
        '3️⃣ *Status do meu pedido* 📦\n'
        '4️⃣ *Falar com um atendente (humano)* 👤'
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


def get_category_menu_text() -> str:
    return (
        'Aqui na Bora Treinar, temos as melhores marcas para o seu objetivo. '
        'Selecione a categoria que você está procurando (Digite o número):\n\n'
        '1️⃣ Proteínas (Whey Concentrado, Isolado, Misto, Albumina) 🥩\n'
        '2️⃣ Aminoácidos (Creatina, BCAA, Glutamina) ⚡\n'
        '3️⃣ Pré-treinos e Emagrecedores (Termogênicos, Energia) 🔥\n'
        '4️⃣ Vitaminas e Saúde (Multivitamínicos, Ômega 3) 💊\n'
        '5️⃣ Combos Promocionais (O melhor custo-benefício!) 🎁\n\n'
        '0️⃣ Voltar ao menu inicial'
    )


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


def get_payment_prompt_text() -> str:
    return (
        '💳 Qual será a forma de pagamento na entrega?\n'
        'Responda com o número:\n'
        '1 - Pix\n'
        '2 - Cartão\n'
        '3 - Dinheiro'
    )


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


def _extract_category_choice(value: str) -> int | None:
    numeric_choice = _extract_numeric_choice(value)
    if numeric_choice in {0, 1, 2, 3, 4, 5}:
        return numeric_choice

    normalized = _normalize_text(value)
    if not normalized:
        return None

    if any(token in normalized for token in {'voltar', 'menu inicial', 'inicio', 'início'}):
        return 0
    if any(token in normalized for token in {'proteina', 'proteínas', 'whey', 'wey', 'albumina'}):
        return 1
    if any(token in normalized for token in {'amino', 'creatina', 'bcaa', 'glutamina'}):
        return 2
    if any(token in normalized for token in {'pre treino', 'pré treino', 'termogenico', 'termogênico', 'emagrecedor'}):
        return 3
    if any(token in normalized for token in {'vitamina', 'multivitaminico', 'multivitamínico', 'omega', 'ômega'}):
        return 4
    if any(token in normalized for token in {'combo', 'promocional', 'kit', 'coqueteleira'}):
        return 5

    return None


def _extract_menu_choice(value: str) -> int | None:
    numeric_choice = _extract_numeric_choice(value)
    if numeric_choice in {1, 2, 3, 4}:
        return numeric_choice

    normalized = _normalize_text(value)
    if not normalized:
        return None

    if 'pedido' in normalized or 'catalogo' in normalized or 'catálogo' in normalized:
        return 1
    if (
        'duvida' in normalized
        or 'dúvida' in normalized
        or 'suplemento' in normalized
        or 'whey' in normalized
        or 'creatina' in normalized
        or 'bcaa' in normalized
    ):
        return 2
    if 'status' in normalized:
        return 3
    if 'atendente' in normalized or 'humano' in normalized:
        return 4

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


def build_cart_summary(cart_items: list[dict]) -> tuple[str, float]:
    if not cart_items:
        return 'Carrinho vazio.', 0.0

    lines = []
    total = 0.0

    for index, item in enumerate(cart_items, start=1):
        price = float(item.get('price', 0))
        quantity = int(item.get('quantity', 0))
        line_total = price * quantity
        total += line_total

        product_name = item.get('product_name') or item.get('name') or 'Produto'
        variation_name = item.get('variation') or 'Unico'

        lines.append(
            f"{index}) Produto: {product_name} | Variacao: {variation_name} | "
            f"Qtd: {quantity} | Preco: {format_brl(price)} | Total: {format_brl(line_total)}"
        )

    return '\n'.join(lines), total


def build_checkout_summary(cart_items: list[dict], address: str, payment_method: str) -> tuple[str, float]:
    items_text, total = build_cart_summary(cart_items)
    text = (
        f'🛒 *Resumo Final*\n'
        f'{items_text}\n\n'
        f'Total: *{format_brl(total)}*\n'
        f'Endereco: {address}\n'
        f'Pagamento: {payment_method}'
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


async def _save_chat_runtime_context(chat_id: str, tenant_id: str | None = None, instance_name: str | None = None):
    if tenant_id:
        await redis_client.set(_tenant_key(chat_id), tenant_id, ex=FLOW_CONTEXT_TTL)
    if instance_name:
        await redis_client.set(_instance_key(chat_id), instance_name, ex=FLOW_CONTEXT_TTL)


async def _get_tenant_id(chat_id: str) -> str:
    tenant_id = await redis_client.get(_tenant_key(chat_id))
    return (tenant_id or '').strip()


async def _get_instance_name(chat_id: str) -> str | None:
    instance_name = await redis_client.get(_instance_key(chat_id))
    instance_name = (instance_name or '').strip()
    return instance_name or None


async def _send_message(chat_id: str, text: str):
    instance_name = await _get_instance_name(chat_id)
    await asyncio.to_thread(
        send_whatsapp_message,
        chat_id,
        text,
        instance_name,
    )


async def _send_message_sequence(chat_id: str, messages: list[str]):
    for idx, msg in enumerate(messages):
        text = str(msg or '').strip()
        if not text:
            continue
        if idx > 0:
            await asyncio.sleep(1.5)
        try:
            await _send_message(chat_id, text)
        except Exception as error:
            log(f'[SEND] Falha ao enviar mensagem {idx + 1} para {chat_id}: {error}')
            continue


async def _send_presence(chat_id: str, presence: str = 'composing', delay: int = 300):
    instance_name = await _get_instance_name(chat_id)
    await asyncio.to_thread(
        send_whatsapp_presence,
        chat_id,
        presence,
        delay,
        instance_name,
    )


async def _send_image(chat_id: str, file_path: str, caption: str = ''):
    instance_name = await _get_instance_name(chat_id)
    await asyncio.to_thread(
        send_whatsapp_image_file,
        chat_id,
        file_path,
        caption,
        instance_name,
    )


async def _send_media(chat_id: str, caption: str, media_url: str):
    instance_name = await _get_instance_name(chat_id)
    await asyncio.to_thread(
        send_whatsapp_media,
        chat_id,
        caption,
        media_url,
        'image',
        None,
        None,
        instance_name,
    )


def _normalize_text(value: str) -> str:
    return value.strip().lower()


async def _load_grouped_products(tenant_id: str) -> dict[str, dict]:
    try:
        grouped_products = await asyncio.to_thread(list_estoque, tenant_id)
    except Exception as error:
        log(f'Erro ao carregar estoque no banco: {error}')
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


async def _send_welcome_sequence(chat_id: str):
    try:
        await _send_message(
            chat_id,
            '💪 Bem-vindo(a) ao atendimento da Bora Treinar Suplementos! O seu parceiro na busca pelos melhores resultados. 🏆',
        )
    except Exception as error:
        log(f'Falha ao enviar boas-vindas para {chat_id}: {error}')

    if not WELCOME_IMAGE_PATH.exists():
        log(f'Imagem de boas-vindas não encontrada em {WELCOME_IMAGE_PATH}')
        return

    try:
        await _send_image(chat_id, str(WELCOME_IMAGE_PATH), '')
    except Exception as error:
        log(f'Falha ao enviar imagem de boas-vindas para {chat_id}: {error}')


async def _restart_flow(chat_id: str):
    await clear_order_context(chat_id)
    await _save_context(
        chat_id,
        {
            'customer_name': '',
            'customer_address': '',
            'payment_method': '',
            'selected_parent_code': '',
            'selected_quantity': '',
            'cart_items': json.dumps([]),
        },
    )
    await _set_state(chat_id, STATE_MENU_INICIAL)
    await _send_welcome_sequence(chat_id)
    return get_main_menu_text()


async def _send_catalog_and_transition(chat_id: str, tenant_id: str) -> str:
    if CATALOG_IMAGE_PATH.exists():
        try:
            await _send_image(chat_id, str(CATALOG_IMAGE_PATH), '🛍️ Confira nosso catálogo!')
        except Exception as error:
            log(f'Falha ao enviar imagem do catálogo para {chat_id}: {error}')

    grouped_products = await _load_grouped_products(tenant_id)
    all_products = _flatten_products(grouped_products)
    if not all_products:
        await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)
        return '📦 No momento estamos sem produtos disponíveis no estoque. Tente novamente em instantes.'

    await _set_state(chat_id, STATE_ESCOLHENDO_CATEGORIA)
    return get_category_menu_text()


async def _send_catalog_for_products(chat_id: str, category_name: str, products: dict[str, dict]) -> str:
    product_list = list(products.values())
    if not product_list:
        await _set_state(chat_id, STATE_ESCOLHENDO_CATEGORIA)
        return 'Essa categoria está sem produtos no momento. Escolha outra categoria.'

    first_product = product_list[0]
    first_image_url = str(first_product.get('imagem_url', '')).strip()
    if _is_valid_media_url(first_image_url):
        try:
            await _send_media(chat_id, f'🔥 Destaques de {category_name}', first_image_url)
        except Exception as error:
            log(f'Falha ao enviar imagem da categoria: {error}')

    await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)
    return get_catalog_text(products)


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


async def _move_to_upsell(chat_id: str, customer_name: str, customer_address: str, cart_items: list[dict]) -> str:
    await _save_context(
        chat_id,
        {
            'customer_name': customer_name,
            'customer_address': customer_address,
            'cart_items': json.dumps(cart_items),
            'selected_parent_code': '',
            'selected_quantity': '',
        },
    )
    await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)

    summary, total = build_cart_summary(cart_items)
    return _with_quick_commands(
        f'✅ Item adicionado ao carrinho.\n\n'
        f'🛒 *Carrinho atual:*\n{summary}\n\n'
        f'Total parcial: *{format_brl(total)}*\n\n'
        'Envie outro código para continuar, *carrinho* para revisar ou *finalizar* para fechar.'
    )


async def _move_to_upsell_with_message(
    chat_id: str,
    customer_name: str,
    customer_address: str,
    cart_items: list[dict],
    intro_message: str,
) -> str:
    await _save_context(
        chat_id,
        {
            'customer_name': customer_name,
            'customer_address': customer_address,
            'cart_items': json.dumps(cart_items),
            'selected_parent_code': '',
            'selected_quantity': '',
        },
    )
    await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)

    summary, total = build_cart_summary(cart_items)
    return _with_quick_commands(
        f'{intro_message}\n\n'
        f'🛒 *Carrinho atual:*\n{summary}\n\n'
        f'Total parcial: *{format_brl(total)}*\n\n'
        'Envie outro código para continuar, *carrinho* para revisar ou *finalizar* para fechar.'
    )


async def route_sales_flow(chat_id: str, user_message: str) -> str:
    normalized_message = _normalize_text(user_message)

    state = await _get_state(chat_id) or STATE_VERIFICACAO_INICIAL
    tenant_id = await _get_tenant_id(chat_id)
    if not tenant_id:
        log(f'Tenant ausente para {chat_id} ao processar fluxo de vendas.')
        return (
            '⚠️ Não consegui identificar sua loja agora. '
            'Envie uma nova mensagem em alguns segundos para tentarmos novamente.'
        )

    if state == STATE_ATENDIMENTO_HUMANO:
        return ''

    if state != STATE_ATENDIMENTO_HUMANO and normalized_message in {'menu'}:
        await _set_state(chat_id, STATE_MENU_INICIAL)
        return get_main_menu_text()

    if normalized_message in {'/cancelar', '/cancel', 'cancelar', 'cancel'}:
        return await _restart_flow(chat_id)

    context = await _get_context(chat_id)
    cart_items = json.loads(context.get('cart_items', '[]'))
    customer_name = context.get('customer_name', '').strip()
    customer_address = context.get('customer_address', '').strip()

    if state == STATE_VERIFICACAO_INICIAL:
        await _save_context(
            chat_id,
            {
                'customer_name': customer_name,
                'customer_address': customer_address,
                'cart_items': json.dumps(cart_items),
            },
        )
        await _set_state(chat_id, STATE_MENU_INICIAL)
        await _send_welcome_sequence(chat_id)
        return get_main_menu_text()

    if state == STATE_MENU_INICIAL:
        menu_choice = _extract_menu_choice(normalized_message)

        if menu_choice == 1:
            return await _send_catalog_and_transition(chat_id)

        if menu_choice == 2:
            await _set_state(chat_id, STATE_DUVIDAS_SUPLEMENTOS)
            return (
                '🤔 Manda sua dúvida sobre suplementos que eu te ajudo agora.\n'
                'Ex.: "qual a diferença entre whey concentrado e isolado?"\n\n'
                'Se quiser voltar, digite *menu*.'
            )

        if menu_choice == 3:
            try:
                ultimo_pedido = await asyncio.to_thread(get_ultimo_pedido, chat_id, tenant_id)
            except Exception as error:
                log(f'Erro ao buscar último pedido no banco: {error}')
                return '⚠️ Não consegui consultar seu último pedido agora.\n\n' + get_main_menu_text()

            if not ultimo_pedido:
                return '📦 Você ainda não possui pedidos recentes por aqui.\n\n' + get_main_menu_text()

            status = ultimo_pedido.get('status') or 'sem status'
            itens = ultimo_pedido.get('itens_resumo') or 'Itens não informados'
            return (
                '📦 *Status do seu último pedido:*\n'
                f'*Status:* {status}\n'
                f'*Itens:*\n{itens}\n\n'
                f'{get_main_menu_text()}'
            )

        if menu_choice == 4:
            await _set_state(chat_id, STATE_ATENDIMENTO_HUMANO)
            return '🤝 Perfeito. Já sinalizei um atendente humano para te chamar em instantes.'

        return 'Não entendi a opção. Responda com 1, 2, 3 ou 4.\n\n' + get_main_menu_text()

    if state == STATE_DUVIDAS_SUPLEMENTOS:
        if 'whey' in normalized_message or 'wey' in normalized_message:
            return get_whey_guidance_text()

        try:
            answer = await asyncio.to_thread(invoke_rag_chain, user_message, chat_id)
        except Exception as error:
            log(f'Erro ao responder dúvida com RAG: {error}')
            return (
                '⚠️ Não consegui consultar as informações agora. Tente novamente em instantes '
                'ou digite *menu* para voltar.'
            )

        if not str(answer).strip():
            return 'Não encontrei essa informação agora. Reformule sua dúvida ou digite *menu* para voltar.'

        return f'{answer}\n\nSe quiser, pode mandar outra dúvida ou digitar *menu* para voltar.'

    if state == STATE_CATALOGO:
        return await _send_catalog_and_transition(chat_id, tenant_id)

    if state == STATE_ESCOLHENDO_CATEGORIA:
        if normalized_message in {'catalogo', 'catálogo'}:
            grouped_products = await _load_grouped_products(tenant_id)
            categories = _fixed_category_options(grouped_products)
            if not any(category.get('produtos') for category in categories):
                return '📦 No momento estamos sem produtos disponíveis no estoque. Tente novamente em instantes.'
            return get_category_menu_text()

        if 'whey' in normalized_message or 'wey' in normalized_message:
            await _set_state(chat_id, STATE_DUVIDAS_SUPLEMENTOS)
            return get_whey_guidance_text()

        choice = _extract_category_choice(normalized_message)
        if choice is None:
            return 'Responda com o número da categoria desejada (ou escreva o nome, ex.: proteínas).'

        grouped_products = await _load_grouped_products(tenant_id)
        categories = _fixed_category_options(grouped_products)

        if not any(category.get('produtos') for category in categories):
            return '📦 No momento estamos sem produtos disponíveis no estoque. Tente novamente em instantes.'

        if choice == 0:
            await _set_state(chat_id, STATE_MENU_INICIAL)
            return get_main_menu_text()

        if choice < 1 or choice > len(categories):
            return (
                f'Opção inválida. Escolha um número de 0 a {len(categories)}.\n\n'
                f'{get_category_menu_text()}'
            )

        selected = categories[choice - 1]
        return await _send_catalog_for_products(
            chat_id,
            selected['categoria'],
            selected['produtos'],
        )

    if state == STATE_ADICIONANDO_CARRINHO:
        if normalized_message in {'catalogo', 'catálogo', 'menu'}:
            return await _send_catalog_and_transition(chat_id, tenant_id)

        if normalized_message == 'carrinho':
            summary, total = build_cart_summary(cart_items)
            return _with_quick_commands(
                f'🛒 *Seu carrinho:*\n{summary}\n\nTotal: *{format_brl(total)}*'
            )

        if normalized_message == 'finalizar':
            if not cart_items:
                return 'Seu carrinho está vazio. Envie um código para adicionar um produto. 💊'

            customer_name, customer_address = await _get_checkout_customer(chat_id, context, tenant_id)

            await _save_context(
                chat_id,
                {
                    'customer_name': customer_name,
                    'customer_address': customer_address,
                    'cart_items': json.dumps(cart_items),
                },
            )

            if not customer_name:
                await _set_state(chat_id, STATE_CHECKOUT_NOME)
                return '🧾 Para finalizar, me diga seu *nome completo*.'

            if not customer_address:
                await _set_state(chat_id, STATE_CHECKOUT_ENDERECO)
                return '📦 Agora me informe seu *endereço completo* para entrega.'

            await _set_state(chat_id, STATE_CHECKOUT_PAGAMENTO)
            return get_payment_prompt_text()

        grouped_products = await _load_grouped_products(tenant_id)
        product_map = _grouped_products_by_code(grouped_products)
        if not product_map:
            return '📦 No momento estamos sem produtos disponíveis no estoque. Tente novamente em instantes.'

        parser_catalog = _catalog_for_intent_parser(product_map)

        # ── NLU: Gemini analisa intenção, produto e dados faltantes ──────────
        nlu = await analyze_message(user_message, cart_items, parser_catalog)
        intencao = nlu.get('intencao', '')
        log(f'NLU para {chat_id}: intencao={intencao!r}')

        # ── 1. Dúvida técnica → RAG (ChromaDB + OpenAI) ──────────────────────
        if intencao == 'duvida_tecnica':
            try:
                answer = await asyncio.to_thread(invoke_rag_chain, user_message, chat_id)
                return answer or await asyncio.to_thread(
                    generate_persona_response,
                    f'O usuário perguntou "{user_message}". Informe que não encontrou a resposta e ofereça atendimento humano.',
                    user_message,
                    chat_id,
                )
            except Exception as err:
                log(f'Erro ao chamar RAG chain: {err}')
                await _set_state(chat_id, STATE_ATENDIMENTO_HUMANO)
                return await asyncio.to_thread(
                    generate_persona_response,
                    'Não consegui consultar as informações agora. Direcione o cliente para um atendente humano.',
                    user_message,
                    chat_id,
                )

        # ── 2. Ver carrinho via linguagem natural ─────────────────────────────
        if intencao == 'ver_carrinho':
            summary, total = build_cart_summary(cart_items)
            instruction = (
                f'Mostre o carrinho do cliente de forma animada. '
                f'Itens: {summary}. Total: {format_brl(total)}. '
                'Oriente a enviar "finalizar" para fechar o pedido ou continue adicionando itens.'
            )
            return await asyncio.to_thread(generate_persona_response, instruction, user_message, chat_id)

        # ── 3. Checkout via linguagem natural ─────────────────────────────────
        if intencao == 'checkout':
            if not cart_items:
                return await asyncio.to_thread(
                    generate_persona_response,
                    'O cliente quer finalizar, mas o carrinho está vazio. Peça para adicionar produtos primeiro.',
                    user_message,
                    chat_id,
                )
            customer_name, customer_address = await _get_checkout_customer(chat_id, context, tenant_id)
            await _save_context(
                chat_id,
                {
                    'customer_name': customer_name,
                    'customer_address': customer_address,
                    'cart_items': json.dumps(cart_items),
                },
            )
            if not customer_name:
                await _set_state(chat_id, STATE_CHECKOUT_NOME)
                return await asyncio.to_thread(
                    generate_persona_response,
                    'O cliente quer finalizar o pedido. Peça o nome completo de forma animada.',
                    user_message,
                    chat_id,
                )
            if not customer_address:
                await _set_state(chat_id, STATE_CHECKOUT_ENDERECO)
                return await asyncio.to_thread(
                    generate_persona_response,
                    'O cliente quer finalizar. Peça o endereço completo para entrega.',
                    user_message,
                    chat_id,
                )
            await _set_state(chat_id, STATE_CHECKOUT_PAGAMENTO)
            return await asyncio.to_thread(
                generate_persona_response,
                'Pergunte a forma de pagamento. Opções: 1 - Pix, 2 - Cartão, 3 - Dinheiro.',
                user_message,
                chat_id,
            )

        # ── 4. Solicitação de atendimento humano ──────────────────────────────
        if intencao == 'atendimento_humano':
            await _set_state(chat_id, STATE_ATENDIMENTO_HUMANO)
            return await asyncio.to_thread(
                generate_persona_response,
                'O cliente pediu atendimento humano. Informe que um atendente entrará em contato em breve.',
                user_message,
                chat_id,
            )

        # ── 5. Adicionar ao carrinho ──────────────────────────────────────────
        if intencao == 'adicionar_carrinho':
            produto = nlu.get('produto_identificado') or {}
            status_item = nlu.get('status_item', '')
            dados_faltantes = nlu.get('dados_faltantes') or []
            upsell_sugerido = nlu.get('upsell_sugerido')

            # 5a. Produto incompleto: pedir dados faltantes via persona
            if status_item == 'incompleto':
                categoria = produto.get('categoria') or 'produto'
                faltantes_str = ', '.join(dados_faltantes) if dados_faltantes else 'mais detalhes (marca, sabor ou tamanho)'
                instruction = (
                    f'O cliente quer {categoria}, mas ainda faltam: {faltantes_str}. '
                    'Pergunte de forma animada e objetiva quais informações estão faltando.'
                )
                return await asyncio.to_thread(generate_persona_response, instruction, user_message, chat_id)

            # 5b. Produto completo: adicionar ao carrinho e confirmar via persona
            if status_item == 'completo':
                codigo_pai = str(produto.get('codigo_pai') or '').strip().lower()
                variacao_name = str(produto.get('variacao') or '').strip()
                try:
                    quantidade = max(1, int(produto.get('quantidade', 1)))
                except (TypeError, ValueError):
                    quantidade = 1

                if not codigo_pai:
                    instruction = (
                        f'O produto "{produto.get("categoria", "item")} {produto.get("marca", "")}" '
                        'não foi encontrado no catálogo. Oriente o cliente a ver o catálogo disponível.'
                    )
                    return await asyncio.to_thread(generate_persona_response, instruction, user_message, chat_id)

                product_group = product_map.get(codigo_pai)
                if not product_group:
                    instruction = (
                        f'O produto com código "{codigo_pai}" não está disponível. '
                        'Oriente o cliente a ver o catálogo.'
                    )
                    return await asyncio.to_thread(generate_persona_response, instruction, user_message, chat_id)

                chosen_variation = _find_catalog_variation(product_group, variacao_name)
                if not chosen_variation:
                    return get_variation_options_text(product_group)

                _add_item_to_cart(cart_items, product_group, chosen_variation, quantidade)

                chosen_variation_label = str(chosen_variation.get('variacao', '')).strip() or 'Unico'
                upsell_str = f' Sugira também: {upsell_sugerido}.' if upsell_sugerido else ''
                instruction = (
                    f'Confirme que adicionamos {product_group.get("nome_produto", "")} '
                    f'({chosen_variation_label}) x{quantidade} ao carrinho.{upsell_str} '
                    'Seja animado. Não liste o carrinho completo aqui, apenas confirme o item adicionado.'
                )
                intro = await asyncio.to_thread(generate_persona_response, instruction, user_message, chat_id)
                return await _move_to_upsell_with_message(
                    chat_id,
                    customer_name,
                    customer_address,
                    cart_items,
                    intro,
                )

        # ── 6. Fallback: intenção não reconhecida ─────────────────────────────
        instruction = (
            f'O cliente enviou uma mensagem não reconhecida: "{user_message}". '
            'Oriente-o de forma amigável a informar se quer adicionar um produto, '
            'ver o carrinho, finalizar o pedido ou falar com um atendente.'
        )
        return await asyncio.to_thread(generate_persona_response, instruction, user_message, chat_id)

    if state == STATE_AGUARDANDO_VARIACAO:
        choice_index = _extract_numeric_choice(normalized_message)
        if choice_index is None:
            return 'Responda apenas com o número da variação desejada. Exemplo: 1.'

        selected_parent_code = context.get('selected_parent_code', '').strip()
        selected_quantity_raw = context.get('selected_quantity', '1').strip()

        if not selected_parent_code:
            await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)
            return 'Não encontrei o produto em seleção. Envie o código novamente para continuar.'

        try:
            selected_quantity = int(selected_quantity_raw)
        except ValueError:
            selected_quantity = 1

        if selected_quantity <= 0:
            selected_quantity = 1

        grouped_products = await _load_grouped_products(tenant_id)
        product_map = _grouped_products_by_code(grouped_products)
        product_group = product_map.get(selected_parent_code.lower())

        if not product_group:
            await _save_context(
                chat_id,
                {
                    'customer_name': customer_name,
                    'customer_address': customer_address,
                    'cart_items': json.dumps(cart_items),
                    'selected_parent_code': '',
                    'selected_quantity': '',
                },
            )
            await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)
            return 'Esse produto não está mais disponível. Escolha outro código no catálogo.'

        variations = product_group.get('variacoes', [])
        if choice_index < 1 or choice_index > len(variations):
            return f'Opção inválida. Escolha um número de 1 a {len(variations)}.'

        chosen_variation = variations[choice_index - 1]
        _add_item_to_cart(cart_items, product_group, chosen_variation, selected_quantity)

        return await _move_to_upsell(chat_id, customer_name, customer_address, cart_items)

    if state == STATE_CHECKOUT_NOME:
        if len(user_message.strip()) < 2:
            return 'Nome inválido. Informe um nome com pelo menos 2 caracteres.'

        customer_name = user_message.strip()
        await _save_context(
            chat_id,
            {
                'customer_name': customer_name,
                'customer_address': customer_address,
                'cart_items': json.dumps(cart_items),
            },
        )

        if not customer_address:
            await _set_state(chat_id, STATE_CHECKOUT_ENDERECO)
            return '📦 Agora me informe seu *endereço completo* para entrega.'

        await _set_state(chat_id, STATE_CHECKOUT_PAGAMENTO)
        return get_payment_prompt_text()

    if state == STATE_CHECKOUT_ENDERECO:
        if len(user_message.strip()) < 5:
            return 'Endereço muito curto. Informe rua e número, por favor.'

        customer_address = user_message.strip()
        await _save_context(
            chat_id,
            {
                'customer_name': customer_name,
                'customer_address': customer_address,
                'cart_items': json.dumps(cart_items),
            },
        )

        if not customer_name:
            await _set_state(chat_id, STATE_CHECKOUT_NOME)
            return '🧾 Para finalizar, me diga seu *nome completo*.'

        await _set_state(chat_id, STATE_CHECKOUT_PAGAMENTO)
        return get_payment_prompt_text()

    if state == STATE_CHECKOUT_PAGAMENTO:
        payment_choice = _extract_numeric_choice(normalized_message)
        payment_method = _payment_label(str(payment_choice)) if payment_choice is not None else None
        if not payment_method:
            return 'Opção inválida. Responda com 1 (Pix), 2 (Cartão) ou 3 (Dinheiro).'

        await _save_context(
            chat_id,
            {
                'customer_name': customer_name,
                'customer_address': customer_address,
                'payment_method': payment_method,
                'cart_items': json.dumps(cart_items),
            },
        )
        await _set_state(chat_id, STATE_FINALIZACAO_CONFIRMACAO)

        final_summary, _ = build_checkout_summary(cart_items, customer_address, payment_method)
        return (
            f'{final_summary}\n\n'
            '✅ Confirma o pedido? Responda *SIM* para confirmar ou *NÃO* para voltar ao carrinho.'
        )

    if state == STATE_FINALIZACAO_CONFIRMACAO:
        decision = _parse_yes_no(normalized_message)
        if not decision:
            return 'Responda *SIM* para confirmar ou *NÃO* para voltar ao carrinho.'

        if decision == 'no':
            await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)
            summary, total = build_cart_summary(cart_items)
            return _with_quick_commands(
                f'Perfeito, voltamos para o carrinho.\n\n'
                f'🛒 *Carrinho:*\n{summary}\n\n'
                f'Total: *{format_brl(total)}*\n\n'
                'Envie outro código, *carrinho* ou *finalizar*.'
            )

        payment_method = context.get('payment_method', '').strip()
        if not payment_method:
            await _set_state(chat_id, STATE_CHECKOUT_PAGAMENTO)
            return get_payment_prompt_text()

        if not customer_name:
            await _set_state(chat_id, STATE_CHECKOUT_NOME)
            return '🧾 Antes de concluir, me diga seu *nome completo*.'

        if not customer_address:
            await _set_state(chat_id, STATE_CHECKOUT_ENDERECO)
            return '📦 Antes de concluir, me informe seu *endereço completo*.'

        summary, total = build_cart_summary(cart_items)

        try:
            pedido = await save_order(
                tenant_id=tenant_id,
                phone=chat_id,
                nome=customer_name,
                endereco=customer_address,
                cart_items=cart_items,
                total=total,
                forma_pagamento=payment_method,
            )
        except Exception as error:
            log(f'Erro ao criar pedido no banco: {error}')
            return '⚠️ Não consegui registrar seu pedido agora. Tente confirmar novamente em instantes.'

        if ADMIN_WHATSAPP_NUMBER:
            try:
                admin_message = _build_admin_summary(pedido)
                instance_name = await _get_instance_name(chat_id)
                await asyncio.to_thread(
                    send_whatsapp_message,
                    ADMIN_WHATSAPP_NUMBER,
                    admin_message,
                    instance_name,
                )
            except Exception as error:
                log(f'Falha ao notificar administrador: {error}')

        await _save_context(
            chat_id,
            {
                'customer_name': customer_name,
                'customer_address': customer_address,
                'payment_method': '',
                'selected_parent_code': '',
                'selected_quantity': '',
            },
        )
        await clear_cart(chat_id)
        await _set_state(chat_id, STATE_MENU_INICIAL)

        final_summary, _ = build_checkout_summary(cart_items, customer_address, payment_method)
        return (
            '✅ *Pedido finalizado com sucesso!*\n\n'
            f'{final_summary}\n\n'
            '📦 Seu pedido foi enviado para separação.\n'
            'Se quiser continuar comprando, responda com uma opção do menu.\n\n'
            f'{get_main_menu_text()}'
        )

    await _set_state(chat_id, STATE_VERIFICACAO_INICIAL)
    return 'Vamos reiniciar seu atendimento. Envie uma mensagem para começar.'


async def buffer_message(
    chat_id: str,
    message: str,
    tenant_id: str | None = None,
    instance_name: str | None = None,
):
    buffer_key = f'{chat_id}{BUFFER_KEY_SUFIX}'

    await _save_chat_runtime_context(
        chat_id=chat_id,
        tenant_id=(tenant_id or '').strip() or None,
        instance_name=(instance_name or '').strip() or None,
    )

    await redis_client.rpush(buffer_key, message)
    await redis_client.expire(buffer_key, BUFFER_TTL)

    log(f'Mensagem adicionada ao buffer de {chat_id}: {message}')

    if debounce_tasks.get(chat_id):
        debounce_tasks[chat_id].cancel()
        log(f'Debounce resetado para {chat_id}')

    debounce_tasks[chat_id] = asyncio.create_task(handle_debounce(chat_id))


async def handle_debounce(chat_id: str):
    buffer_key = f'{chat_id}{BUFFER_KEY_SUFIX}'

    try:
        log(f'Iniciando debounce para {chat_id}')
        await asyncio.sleep(float(DEBOUNCE_SECONDS))

        messages = await redis_client.lrange(buffer_key, 0, -1)

        full_message = ' '.join(messages).strip()
        if full_message:
            log(f'Enviando mensagem agrupada para {chat_id}: {full_message}')
            reply_message = await route_sales_flow(chat_id, full_message)

            if not reply_message:
                await redis_client.delete(buffer_key)
                return

            try:
                await _send_presence(chat_id, presence='composing', delay=300)
            except Exception as error:
                log(f'Falha ao enviar presença para {chat_id}: {error}')

            try:
                if isinstance(reply_message, list):
                    await _send_message_sequence(chat_id, reply_message)
                else:
                    await _send_message(chat_id, str(reply_message))
            except Exception as error:
                log(f'Falha ao enviar mensagem para {chat_id}: {error}')
        await redis_client.delete(buffer_key)

    except asyncio.CancelledError:
        log(f'Debounce cancelado para {chat_id}')
    except Exception as error:
        log(f'Erro inesperado no debounce de {chat_id}: {error}')
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
            await _send_message(chat_id, fallback_message)
        except Exception as send_error:
            log(f'Falha ao enviar fallback para {chat_id}: {send_error}')
        try:
            await redis_client.delete(buffer_key)
        except Exception as delete_error:
            log(f'Falha ao limpar buffer de {chat_id}: {delete_error}')
    finally:
        current_task = debounce_tasks.get(chat_id)
        running_task = asyncio.current_task()
        if current_task is running_task:
            debounce_tasks.pop(chat_id, None)
