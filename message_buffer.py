import asyncio
import json
import re
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
)
from chains import generate_persona_response, invoke_rag_chain
from gemini_parser import analyze_message, extract_order_intent


redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
debounce_tasks = defaultdict(asyncio.Task)

FLOW_STATE_SUFFIX = '_flow_state'
FLOW_CONTEXT_SUFFIX = '_flow_context'

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


def get_main_menu_text() -> str:
    return (
        'Como posso te ajudar a focar no treino hoje? (Digite o número da opção desejada):\n\n'
        '1️⃣ *Fazer um pedido / Ver Catálogo* 🛒\n'
        '2️⃣ *Dúvidas sobre suplementos (Whey, Creatina, Pré-treino...)* 🤔\n'
        '3️⃣ *Status do meu pedido* 📦\n'
        '4️⃣ *Falar com um atendente (humano)* 👤'
    )


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
    payment = payment.replace('✅', '').strip()
    if '\n' in payment:
        payment = payment.splitlines()[0].strip()
    return items, address, payment


def _build_cart_items_from_summary(items_text: str) -> list[dict]:
    normalized_items = (items_text or '').strip()
    if not normalized_items:
        return []

    quantity_match = re.match(r'^\s*(\d+)\s*[xX]\s+(.+)$', normalized_items)
    if quantity_match:
        quantity = int(quantity_match.group(1))
        product_name = quantity_match.group(2).strip()
    else:
        quantity = 1
        product_name = normalized_items

    return [
        {
            'product_name': product_name,
            'name': product_name,
            'quantity': max(1, quantity),
            'price': 0.0,
        }
    ]


async def _persist_ai_final_order_if_needed(
    chat_id: str,
    tenant_id: str,
    reply_message: str,
) -> None:
    normalized_reply = (reply_message or '').strip()
    if '*Resumo do Pedido*' not in normalized_reply or not normalized_reply.endswith('✅'):
        return

    items_text, address, payment = _extract_order_summary_fields(normalized_reply)
    if not items_text or not address or not payment:
        log('Resumo final detectado, mas campos obrigatórios incompletos para persistência.')
        return

    context_key = _context_key(chat_id)
    fingerprint = f'{items_text}|{address}|{payment}'.strip().lower()
    last_fingerprint = (await redis_client.hget(context_key, 'last_order_fingerprint') or '').strip().lower()
    if last_fingerprint and last_fingerprint == fingerprint:
        log('Pedido final já persistido anteriormente para este chat; ignorando duplicidade.')
        return

    cart_items = _build_cart_items_from_summary(items_text)
    if not cart_items:
        return

    total = sum(float(item.get('price') or 0) * int(item.get('quantity') or 0) for item in cart_items)

    try:
        await save_order(
            tenant_id=tenant_id,
            phone=chat_id,
            nome='Cliente WhatsApp',
            endereco=address,
            cart_items=cart_items,
            total=total,
            forma_pagamento=payment,
        )
        await redis_client.hset(context_key, mapping={'last_order_fingerprint': fingerprint})
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


async def _restart_flow(chat_id: str, instance_name: str):
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
    await _send_welcome_sequence(chat_id, instance_name)
    return get_main_menu_text()


async def _send_catalog_and_transition(chat_id: str, tenant_id: str, instance_name: str) -> str:
    if CATALOG_IMAGE_PATH.exists():
        try:
            await asyncio.to_thread(
                send_whatsapp_image_file,
                chat_id,
                str(CATALOG_IMAGE_PATH),
                '🛍️ Confira nosso catálogo!',
                instance_name,
            )
        except Exception as error:
            log(f'Falha ao enviar imagem do catálogo para {chat_id}: {error}')

    grouped_products = await _load_grouped_products(tenant_id)
    all_products = _flatten_products(grouped_products)
    if not all_products:
        await _set_state(chat_id, STATE_ADICIONANDO_CARRINHO)
        return '📦 No momento estamos sem produtos disponíveis no estoque. Tente novamente em instantes.'

    await _set_state(chat_id, STATE_ESCOLHENDO_CATEGORIA)
    return get_category_menu_text()


async def _send_catalog_for_products(chat_id: str, category_name: str, products: dict[str, dict], instance_name: str) -> str:
    product_list = list(products.values())
    if not product_list:
        await _set_state(chat_id, STATE_ESCOLHENDO_CATEGORIA)
        return 'Essa categoria está sem produtos no momento. Escolha outra categoria.'

    first_product = product_list[0]
    first_image_url = str(first_product.get('imagem_url', '')).strip()
    if _is_valid_media_url(first_image_url):
        try:
            await asyncio.to_thread(
                send_whatsapp_media,
                chat_id,
                f'🔥 Destaques de {category_name}',
                first_image_url,
                instance_name=instance_name,
            )
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


async def process_message(chat_id: str, user_message: str, tenant_id: str, instance_name: str) -> str:
    # Modo IA pura: sem fluxos de menu/carrinho/checkout.
    conversation_id = f'{tenant_id}:{chat_id}'

    try:
        tenant_configs = await get_tenant_configs(tenant_id)
    except Exception as error:
        log(f'Erro ao carregar configs do tenant no banco: {error}')
        tenant_configs = {
            'promptIa': '',
            'whatsappAdmin': '',
            'botObjective': 'FECHAR_PEDIDO',
        }

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
                await _persist_ai_final_order_if_needed(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    reply_message=reply_message,
                )
            except Exception as error:
                log(f'Falha ao enviar mensagem para {chat_id}: {error}')
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
