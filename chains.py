import re
import json
import asyncio
import random
import traceback
import unicodedata
import logging
import concurrent.futures
import os

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from google import genai
from google.genai import types
import redis as redis_sync
import redis.asyncio as redis_async

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    OPENAI_MODEL_TEMPERATURE,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    REDIS_URL,
)
from database_api import fetch_active_produtos, get_tenant_configs, fetch_stock_for_context, get_tenant_sub_nicho
from memory import get_session_history
from order_extractor import build_order_payload_from_history_window, collect_active_human_texts, slice_messages_after_last_completed_order
from vectorstore import get_vectorstore
from prompts import contextualize_prompt, qa_prompt

logger = logging.getLogger(__name__)

# ─── Redis sync client (usado em thread pool via asyncio.to_thread) ───────────

_STOCK_CACHE_TTL = 300  # 5 minutos

_redis_sync_client: redis_sync.Redis | None = None
_redis_async_client: redis_async.Redis | None = None


def _get_redis_sync() -> redis_sync.Redis | None:
    global _redis_sync_client
    if _redis_sync_client is None and REDIS_URL:
        try:
            _redis_sync_client = redis_sync.Redis.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            pass
    return _redis_sync_client


def _get_redis_async() -> redis_async.Redis | None:
    global _redis_async_client
    if _redis_async_client is None and REDIS_URL:
        try:
            _redis_async_client = redis_async.Redis.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            pass
    return _redis_async_client


def get_rag_chain():
    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        temperature=OPENAI_MODEL_TEMPERATURE,
    )
    retriever = get_vectorstore().as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_conversational_rag_chain():
    rag_chain = get_rag_chain()
    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )


_PERSONA_SYSTEM_PROMPT = (
    'Você é um atendente virtual de WhatsApp. '
    'Seja direto, natural e humano. '
    'Nunca use linguagem robótica ou formal demais. '
    'Máximo 2-3 linhas por mensagem. Nunca liste mais de 5 itens de uma vez.'
)

_NICHO_SYSTEM_PROMPTS: dict[str, str] = {
    'adega': (
        'IDENTIDADE:\n'
        'Você é {nome_atendente}, atendente virtual da {nome_negocio}, uma adega de bairro.\n'
        'Tom: descontraído, jovem, direto. Como um atendente de balcão que conhece os clientes.\n'
        'Use emojis com moderação — só quando adiciona emoção real, não decoração.\n\n'
        'REGRAS DE OURO:\n'
        '1. Máximo 2-3 linhas por mensagem. Jamais escreva parágrafos longos.\n'
        '2. Nunca repita a pergunta de confirmação de produto se o cliente já confirmou.\n'
        '3. Depois que o cliente confirmar o produto, siga direto para endereço.\n'
        '4. Depois que tiver endereço e pagamento, monte o resumo e finalize — sem perguntar de novo.\n'
        '5. Se o cliente mandar "sim", "isso", "pode ser", "exato" — interprete como confirmação, não peça mais detalhes.\n'
        '6. Nunca use colchetes ou placeholders na resposta — use os dados reais ou omita o campo.\n'
        '7. Se não souber algo, diga de forma natural: "Não tenho essa info agora, mas posso te ajudar com o pedido!"\n\n'
        'FLUXO DE PEDIDO (siga essa ordem, sem pular):\n'
        'Passo 1 — Cliente pede produto → confirmar o item se houver ambiguidade (só uma vez)\n'
        'Passo 2 — Produto confirmado → pedir endereço completo\n'
        'Passo 3 — Endereço recebido → pedir forma de pagamento (se não informou junto)\n'
        'Passo 4 — Tudo coletado → montar resumo e finalizar\n\n'
        'ESTILO DE CONFIRMAÇÃO DE PRODUTO (varie entre essas opções):\n'
        '- "Heineken 600ml, certo? 🍺"\n'
        '- "[produto], pode ser?"\n'
        '- "É [produto] mesmo?"\n'
        '- "[produto] então, tô anotando — confirma?"\n\n'
        'APÓS CONFIRMAÇÃO DO CLIENTE:\n'
        'Varie entre:\n'
        '- "Show! Anotado 🍺 Me manda o endereço?"\n'
        '- "Fechado! Me passa o endereço pra entrega?"\n'
        '- "Boa! Vou separar já. Endereço?"\n'
        '- "Perfeito! Entrega ou retirada?"\n\n'
        'NUNCA FAÇA:\n'
        '- Perguntar "você quis dizer X?" depois que o cliente já confirmou\n'
        '- Repetir o resumo mais de uma vez\n'
        '- Usar linguagem formal ("prezado cliente", "conforme solicitado")\n'
        '- Enviar mensagem vazia ou com placeholders como [endereço informado pelo cliente]\n'
        '- "Para confirmar certinho:"\n'
        '- "Se for isso, responde sim, esse"\n'
        '- "Me manda o sabor correto"'
    ),
    'lanchonete': (
        'IDENTIDADE:\n'
        'Você é {nome_atendente}, atendente virtual da {nome_negocio}, uma lanchonete.\n'
        'Tom: animado, simpático e direto. Como um atendente de balcão que quer fechar logo o pedido.\n'
        'Use emojis com moderação.\n\n'
        'REGRAS DE OURO:\n'
        '1. Máximo 2-3 linhas por mensagem.\n'
        '2. Quando o cliente confirmar produto E der quantidade na mesma mensagem ("sim, quero 2"), registre ambos e siga para próximo passo — NÃO pergunte produto de novo.\n'
        '3. Frases como "só isso", "é só isso", "só isso mesmo", "pode fechar", "só isso só" significam pedido completo — NÃO pergunte mais produtos.\n'
        '4. Se o cliente já deu pagamento junto com "só isso" (ex.: "só isso, pix"), registre pagamento e peça endereço.\n'
        '5. Depois de ter produto + pagamento + endereço, monte resumo e finalize — sem perguntar de novo.\n'
        '6. NUNCA pergunte "Qual produto você quer pedir?" mais de uma vez por fluxo.\n'
        '7. Se o cliente repetir o pedido com frustração, peça desculpa brevemente e finalize o passo pendente.\n\n'
        'FLUXO DE PEDIDO:\n'
        'Passo 1 — Cliente pede -> confirmar se ambíguo (só uma vez)\n'
        'Passo 2 — Confirmado -> se cliente disse "só isso" ou similar -> pedir endereço\n'
        'Passo 3 — Se cliente deu pagamento junto -> registrar e pedir só endereço\n'
        'Passo 4 — Endereço recebido -> montar resumo e finalizar\n\n'
        'DETECTAR FECHAMENTO DO PEDIDO:\n'
        'Palavras: "só isso", "é isso", "pode fechar", "fecha o pedido", "só isso mesmo", "só isso só", "por enquanto é isso", "tá bom assim".\n'
        'Quando detectar -> NÃO perguntar mais produtos -> seguir para endereço/pagamento.\n\n'
        'ESTILO DE CONFIRMAÇÃO:\n'
        '- "X-Bacon, pode ser? 🥪"\n'
        '- "É [produto] mesmo?"\n'
        '- "[produto], certo?"\n\n'
        'APÓS CONFIRMAÇÃO:\n'
        '- "Perfeito! [X]x [produto] — R$XX. Endereço pra entrega?"\n'
        '- "Anotado! R$XX no [pagamento]. Me manda o endereço!"\n'
        '- "Show! Vou separar. Pra onde mando?"\n\n'
        'NUNCA FAÇA:\n'
        '- Perguntar "Qual produto você quer pedir?" após o cliente já ter pedido\n'
        '- Ignorar "só isso" e continuar pedindo produtos\n'
        '- Perguntar produto depois de receber endereço'
    ),
}


def get_nicho_prompt(sub_nicho: str, tenant_config: dict) -> str:
    template = _NICHO_SYSTEM_PROMPTS.get((sub_nicho or '').strip().lower(), '')
    if not template:
        return ''

    return template.format(
        nome_atendente=str(tenant_config.get('nome_atendente') or 'Assistente').strip() or 'Assistente',
        nome_negocio=str(tenant_config.get('nome_negocio') or 'nossa loja').strip() or 'nossa loja',
    )


async def get_dynamic_context(
    tenant_id: str,
    phone: str,
    redis_client: redis_async.Redis | None,
) -> str:
    if not redis_client or not tenant_id or not phone:
        return ''

    key = f'session:{tenant_id}:{phone}:produto_confirmado'
    try:
        produto_confirmado = await redis_client.get(key)
    except Exception:
        return ''

    if produto_confirmado:
        return (
            '\nCONTEXTO ATUAL DA CONVERSA:\n'
            '- O cliente JÁ confirmou o produto. NÃO pergunte confirmação de produto novamente.\n'
            '- Foque em coletar endereço e pagamento se ainda não tiver.\n'
            '- Se já tiver tudo, monte o resumo e finalize.\n'
        )
    return ''

_CATALOGO_GOLD_RULE = (
    'Atenção: Você só pode oferecer e vender os produtos que estão listados no CATÁLOGO abaixo. '
    'Respeite rigorosamente as regras de Variações, Extras e Limites descritas no contexto de cada item. '
    'Não invente produtos ou preços.'
)

_ALTERNATIVE_WHEN_UNAVAILABLE_RULE = (
    'REGRA DE CONTINUIDADE COMERCIAL:\n'
    '- Se o cliente pedir algo que não existe no cardápio ou esteja indisponível, NUNCA pare na negativa.\n'
    '- Sempre responda de forma natural e já sugira a alternativa mais próxima disponível.\n'
    '- Exemplo de estilo: "[produto] não tenho hoje, mas tenho [alternativa] que é igualmente incrível. Quer?"\n'
    '- Objetivo: nunca travar a conversa; sempre conduzir para uma opção viável.'
)

_TEST_DRIVE_SUB_NICHO_RULES: dict[str, str] = {
    'adega': (
        'PROMPT DE TEST-DRIVE (ADEGA):\n'
        'Você é o atendente virtual da loja, uma adega de bairro.\n'
        "Cardápio: Cervejas (Heineken 600ml R$12, Brahma 600ml R$11, Corona R$11), "
        "Destilados (Smirnoff 998ml R$49,90, Jack Daniel's 1L R$139,90), "
        'Energéticos (Red Bull 250ml R$9,90), Narguilé (Essência Duas Maçãs R$18, Carvão 1kg R$12), '
        'Petiscos (Amendoim R$6).\n'
        'Combos: Combo Fim de Semana (Smirnoff + 2x Red Bull + Gelo) R$72.\n'
        'Tom: descontraído, jovem, usa emojis com moderação.\n'
        'NUNCA trave numa negativa — sempre sugira alternativa.'
    ),
    'lanchonete': (
        'PROMPT DE TEST-DRIVE (LANCHONETE):\n'
        'Você é o atendente virtual da loja, uma lanchonete.\n'
        'Cardápio: Lanches (X-Burguer R$18, X-Bacon R$22, X-Tudo R$26), '
        'Combos (X-Burguer+fritas+suco R$28, X-Bacon+fritas+refri R$32), '
        'Bebidas (Suco natural R$8, Refri R$5).\n'
        'Tom: animado, simpático, sugere combos com frequência.\n'
        'NUNCA trave numa negativa — sempre sugira alternativa.'
    ),
    'pizzaria': (
        'PROMPT DE TEST-DRIVE (PIZZARIA):\n'
        'Você é o atendente virtual da loja, uma pizzaria.\n'
        'Cardápio: Tradicionais (Calabresa base R$35, Frango Catupiry R$38, Portuguesa R$38), '
        'Especiais (4 Queijos R$42, Frango Bacon R$44).\n'
        'Tamanhos: Pequena (+R$0), Média (+R$10), Grande (+R$20), GG (+R$35).\n'
        'Bordas: sem borda, catupiry (+R$8), cheddar (+R$8).\n'
        'SEMPRE pergunte tamanho e borda antes de confirmar o pedido.\n'
        'Tom: descontraído, guia o cliente pelo pedido passo a passo.\n'
        'NUNCA trave numa negativa — sempre sugira alternativa.'
    ),
}

_WHATSAPP_HARD_RULES = (
    'REGRAS GERAIS (nunca quebre):\n'
    '1) NUNCA envie links externos. Todo atendimento acontece aqui no chat.\n'
    '2) NUNCA use mensagens vazias (só elogio/confirmação). Sempre avance para o próximo dado necessário ou feche.\n'
    '3) FORMATAÇÃO WHATSAPP: use *negrito* para dados importantes e quebras de linha para legibilidade.\n'
    '4) NATURALIDADE: não inicie com "Compreendido", "Entendido", "Entendi" ou "Anotado".\n'
    '5) Use no máximo 1 emoji por mensagem.'
)

_FUNNEL_MODULES_RULES: dict[str, str] = {
    'FECHAR_PEDIDO': (
        'MÓDULO: FECHAR PEDIDO (varejo/alimentação).\n'
        'CHECKLIST OBRIGATÓRIO: 1) item(ns), 2) tamanho/quantidade, 3) endereço completo, 4) pagamento.\n'
        'Se cliente mandar pedido + endereço na mesma frase, peça imediatamente os faltantes (tamanho/quantidade e pagamento).\n'
        'VOCÊ JÁ TEM TODOS OS DADOS? Se sim, responda APENAS com o resumo do pedido formatado para WhatsApp, '
        'liste os itens, endereço e forma de pagamento, e encerre com o emoji ✅. Não diga mais nada.\n'
        'TEMPLATE FINAL OBRIGATÓRIO:\n\n'
        '*Resumo do Pedido*\n'
        '🛒 *Itens:* [lista detalhada com tamanhos]\n'
        '📍 *Endereço:* [endereço completo do cliente]\n'
        '💳 *Pagamento:* [forma escolhida]\n'
        '✅'
    ),
    'AGENDAR': (
        'MÓDULO: AGENDAR HORÁRIO (serviços/clínicas).\n'
        'CHECKLIST OBRIGATÓRIO: 1) serviço desejado, 2) nome do cliente, 3) dia/horário de preferência.\n'
        'TEMPLATE FINAL OBRIGATÓRIO:\n\n'
        '*Agendamento Confirmado!* ✅\n'
        '👤 *Cliente:* [Nome]\n'
        '📅 *Data/Hora:* [Dia e Hora]\n'
        '✨ *Serviço:* [Serviço escolhido]\n'
        '_Obrigado! Te esperamos no horário marcado._'
    ),
    'TIRAR_DUVIDAS': (
        'MÓDULO: TIRAR DÚVIDAS (suporte/institucional).\n'
        'Responda de forma direta e sempre pergunte se há mais alguma dúvida.\n'
        'Quando cliente disser "obrigado", "só isso" ou encerrar conversa, finalize com o template abaixo.\n'
        'TEMPLATE FINAL OBRIGATÓRIO:\n\n'
        '*Atendimento Encerrado* ✅\n'
        '_Espero ter ajudado! Qualquer nova dúvida, é só chamar._'
    ),
}

_HISTORY_WINDOW_SIZE = 20

_ADEGA_CONTEXT_CRITICAL_RULE = (
    'REGRA DE CONTEXTO CRITICA (ADEGA):\n'
    'Quando o cliente responder com quantidade (ex.: "sim, quero 5", "pode ser 3", "quero 2"), '
    'SEMPRE use o ultimo produto confirmado no historico da conversa.\n'
    'NUNCA pergunte "5 de qual produto?" se o produto ja foi mencionado/confirmado antes.\n'
    'O numero se refere ao ultimo produto confirmado no contexto ativo.\n'
)

_LEADING_ROBOTIC_WORDS = re.compile(
    r'^(compreendido|entendido|entendi|anotado|anotei|registrado|perfeito,\s*anotado)\b[\s,:-]*',
    flags=re.IGNORECASE,
)

_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)

_FINAL_MARKERS = {
    'FECHAR_PEDIDO': '✅',
    'AGENDAR': 'agendamento confirmado',
    'TIRAR_DUVIDAS': 'atendimento encerrado',
}


def _is_affirmative_text(text: str) -> bool:
    normalized = (text or '').strip().lower()
    if not normalized:
        return False

    tokens = (
        'sim',
        'isso',
        'isso mesmo',
        'pode ser',
        'exato',
        'confirmo',
        'sim esse',
        'sim esse mesmo',
    )
    return any(token in normalized for token in tokens)


def _is_order_closure_text(text: str) -> bool:
    normalized = (text or '').strip().lower()
    if not normalized:
        return False

    closure_tokens = (
        'so isso',
        'só isso',
        'e isso',
        'é isso',
        'pode fechar',
        'fecha o pedido',
        'so isso mesmo',
        'só isso mesmo',
        'so isso so',
        'só isso só',
        'por enquanto e isso',
        'por enquanto é isso',
        'ta bom assim',
        'tá bom assim',
        'pode ser isso',
    )
    return any(token in normalized for token in closure_tokens)


def _extract_quantity_from_text(text: str) -> int:
    normalized = (text or '').strip().lower()
    if not normalized:
        return 1

    match = re.search(r'\b(\d+)\b', normalized)
    if match:
        return max(1, int(match.group(1)))

    word_map = {
        'um': 1,
        'uma': 1,
        'dois': 2,
        'duas': 2,
        'tres': 3,
        'três': 3,
        'quatro': 4,
        'cinco': 5,
        'seis': 6,
        'sete': 7,
        'oito': 8,
        'nove': 9,
        'dez': 10,
    }
    for word, value in word_map.items():
        if re.search(rf'\b{word}\b', normalized):
            return value

    return 1


def _recent_ai_requested_product_confirmation(history_window) -> bool:
    for message in reversed(list(history_window)[-4:]):
        if getattr(message, 'type', '') != 'ai':
            continue
        content = str(getattr(message, 'content', '') or '').lower()
        if (
            'você quis dizer' in content
            or 'voce quis dizer' in content
            or 'para confirmar certinho' in content
            or 'responde "sim' in content
            or 'mesmo?' in content
            or 'pode ser?' in content
            or 'certo?' in content
        ):
            return True
    return False


def _extract_confirmed_item_from_ai(history_window) -> str:
    for message in reversed(list(history_window)[-6:]):
        if getattr(message, 'type', '') != 'ai':
            continue
        content = str(getattr(message, 'content', '') or '')
        bold_match = re.search(r'\*(\d+\s*[xX]?\s*[^*]+)\*', content)
        if bold_match:
            return bold_match.group(1).strip()

        confirm_match = re.search(
            r'(?i)^\s*([a-zA-Z0-9\sçÇãÃõÕáéíóúÁÉÍÓÚ-]+?),\s*(?:pode ser|certo)\??\s*$',
            content,
        )
        if confirm_match:
            return confirm_match.group(1).strip()

        question_match = re.search(r'(?i)é\s+([a-zA-Z0-9\sçÇãÃõÕáéíóúÁÉÍÓÚ-]+?)\s+mesmo\??', content)
        if question_match:
            return question_match.group(1).strip()

        plain_match = re.search(r'(?i)(\d+\s*[xX]?\s+[a-zA-Z0-9\sçÇãÃõÕáéíóúÁÉÍÓÚ-]+)', content)
        if plain_match:
            return plain_match.group(1).strip()
    return ''


def _extract_recent_quantity_from_humans(history_window) -> int:
    for message in reversed(list(history_window)[-8:]):
        if getattr(message, 'type', '') != 'human':
            continue
        quantity = _extract_quantity_from_text(str(getattr(message, 'content', '') or ''))
        if quantity > 1:
            return quantity
    return 1


def _is_placeholder_value(value: str) -> bool:
    normalized = (value or '').strip()
    if not normalized:
        return True
    return '[' in normalized and ']' in normalized


async def get_cardapio_context(tenant_id: str) -> str:
    """Traduz o catálogo híbrido ativo do tenant para contexto legível ao LLM."""
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        return 'Nenhum tenant_id informado para carregar catálogo.'

    produtos = await fetch_active_produtos(normalized_tenant_id)
    if not produtos:
        # Fallback para tenants de test-drive sem seed em `produtos`.
        tenant_configs = await get_tenant_configs(normalized_tenant_id)
        tenant_prompt = str(tenant_configs.get('promptIa') or '').strip()

        if tenant_prompt:
            marker = '--- INFORMAÇÕES DO SEU NEGÓCIO ---'
            if marker in tenant_prompt:
                business_context = tenant_prompt.split(marker, maxsplit=1)[1].strip()
                if business_context:
                    return (
                        'Catálogo estruturado (produtos ativos) ainda não encontrado para esta loja.\n'
                        'Use temporariamente o contexto textual abaixo como catálogo válido:\n\n'
                        f'{business_context}'
                    )

            return (
                'Catálogo estruturado (produtos ativos) ainda não encontrado para esta loja.\n'
                'Use temporariamente o prompt operacional do lojista para responder com precisão:\n\n'
                f'{tenant_prompt}'
            )

        return 'Nenhum produto ativo encontrado para esta loja no momento.'

    lines: list[str] = []
    lines.append('Produtos ativos da loja:')

    for index, produto in enumerate(produtos, start=1):
        nome = str(produto.get('nome') or '').strip() or 'Produto sem nome'
        categoria = str(produto.get('categoria') or '').strip() or 'Sem categoria'
        classe_negocio = str(produto.get('classe_negocio') or '').strip() or 'generico'
        preco_base_raw = produto.get('preco_base')
        regras_ia = str(produto.get('regras_ia') or '').strip()

        try:
            preco_base = float(preco_base_raw)
            preco_text = f'R$ {preco_base:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
        except (TypeError, ValueError):
            preco_text = str(preco_base_raw or 'não informado')

        config_nicho = produto.get('config_nicho') or {}
        if isinstance(config_nicho, str):
            try:
                config_nicho = json.loads(config_nicho)
            except json.JSONDecodeError:
                config_nicho = {'raw': config_nicho}

        config_pretty = json.dumps(config_nicho, ensure_ascii=False, indent=2)

        lines.append(f'\n### Item {index}')
        lines.append(f'- Nome: {nome}')
        lines.append(f'- Categoria: {categoria}')
        lines.append(f'- Classe do Negócio: {classe_negocio}')
        lines.append(f'- Preço Base: {preco_text}')
        lines.append('- Configurações do Nicho (JSON):')
        lines.append('```json')
        lines.append(config_pretty)
        lines.append('```')
        if regras_ia:
            lines.append(f'- Regras IA do lojista: {regras_ia}')

    return '\n'.join(lines)


def _format_brl(value: float) -> str:
    return f'R$ {value:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def get_stock_context(tenant_id: str) -> str:
    """
    Retorna o contexto de cardapio por sub-nicho para injetar no system prompt.
    Usa Redis para cache com TTL de 5 minutos.
    Chamado de forma síncrona (já que generate_persona_response roda em thread pool).
    """
    cache_key = f'stock:{tenant_id}'
    r = _get_redis_sync()

    if r:
        try:
            cached = r.get(cache_key)
            if cached:
                return str(cached)
        except Exception:
            pass

    try:
        stock_data = asyncio.run(fetch_stock_for_context(tenant_id))
    except Exception as e:
        return f'Falha ao carregar estoque: {e}'

    sub_nicho = str(stock_data.get('sub_nicho') or '').strip().lower()

    if sub_nicho == 'pizzaria':
        sabores: list[dict] = stock_data.get('sabores', [])
        tamanhos: list[dict] = stock_data.get('tamanhos', [])
        bordas: list[dict] = stock_data.get('bordas', [])

        lines: list[str] = ['CARDAPIO ATUAL (PIZZARIA):']

        if sabores:
            lines.append('\nSABORES DISPONIVEIS AGORA:')
            for sabor in sabores:
                categoria = str(sabor.get('categoria') or 'Outros')
                precos = sabor.get('precos') or {}
                parts: list[str] = []
                if isinstance(precos, dict):
                    for sigla, valor in precos.items():
                        try:
                            parts.append(f"{sigla}: {_format_brl(float(valor))}")
                        except Exception:
                            continue
                precos_txt = ' | '.join(parts) if parts else 'precos nao configurados'
                lines.append(f"- {sabor.get('nome', 'Sabor')} ({categoria}) -> {precos_txt}")
        else:
            lines.append('\nSABORES DISPONIVEIS: nenhum no momento.')

        if tamanhos:
            lines.append('\nTAMANHOS:')
            for tamanho in tamanhos:
                try:
                    mod = float(tamanho.get('modificador_preco') or 0)
                except Exception:
                    mod = 0
                mod_txt = 'preco base' if mod == 0 else f"+{_format_brl(mod)}"
                lines.append(
                    f"- {tamanho.get('sigla', '')} ({tamanho.get('nome', 'Tamanho')}) - {tamanho.get('fatias', 0)} fatias - {mod_txt}"
                )

        if bordas:
            lines.append('\nBORDAS:')
            for borda in bordas:
                try:
                    extra = float(borda.get('preco_extra') or 0)
                except Exception:
                    extra = 0
                extra_txt = 'sem acrescimo' if extra == 0 else f"+{_format_brl(extra)}"
                lines.append(f"- {borda.get('nome', 'Borda')} ({extra_txt})")

        context_text = '\n'.join(lines)

        if r:
            try:
                r.set(cache_key, context_text, ex=_STOCK_CACHE_TTL)
            except Exception:
                pass

        return context_text

    items: list[dict] = stock_data.get('items', [])
    combos: list[dict] = stock_data.get('combos', [])

    disponiveis = [i for i in items if i.get('disponivel')]
    esgotados = [i for i in items if not i.get('disponivel')]

    lines: list[str] = ['ESTOQUE ATUAL (ADEGA):']

    if disponiveis:
        lines.append('\nPRODUTOS DISPONÍVEIS AGORA:')
        for item in disponiveis:
            preco_txt = _format_brl(float(item.get('preco') or 0))
            qty = item.get('quantidade', 0)
            lines.append(f"- {item['nome']} — {preco_txt} ({qty} unidades)")
    else:
        lines.append('\nPRODUTOS DISPONÍVEIS: nenhum no momento.')

    if esgotados:
        lines.append('\nPRODUTOS ESGOTADOS (NÃO OFERECER, NÃO CONFIRMAR):')
        for item in esgotados:
            lines.append(f"- {item['nome']}")

    combos_disponiveis = [c for c in combos if c.get('disponivel')]
    combos_esgotados = [c for c in combos if not c.get('disponivel')]

    if combos_disponiveis:
        lines.append('\nCOMBOS DISPONÍVEIS:')
        for combo in combos_disponiveis:
            preco_txt = _format_brl(float(combo.get('preco') or 0))
            desc = combo.get('descricao') or ''
            lines.append(f"- {combo['nome']} — {preco_txt}" + (f' ({desc})' if desc else ''))

    if combos_esgotados:
        lines.append('\nCOMBOS INDISPONÍVEIS (NÃO OFERECER):')
        for combo in combos_esgotados:
            lines.append(f"- {combo['nome']}")

    context_text = '\n'.join(lines)

    if r:
        try:
            r.set(cache_key, context_text, ex=_STOCK_CACHE_TTL)
        except Exception:
            pass

    return context_text


_ADEGA_STOCK_RULES = (
    'REGRA CRÍTICA DE ESTOQUE (ADEGA):\n'
    '- NUNCA ofereça, sugira ou confirme produtos listados em ESGOTADOS.\n'
    '- Se o cliente pedir um produto esgotado, responda: '
    '"Poxa, [produto] acabou hoje! Mas temos [sugerir alternativa disponível]. Quer experimentar?"\n'
    '- Sempre sugira uma alternativa disponível ao negar um produto esgotado.\n'
    '- Combos só podem ser oferecidos se estiverem na lista COMBOS DISPONÍVEIS.'
)

_PIZZARIA_PRICE_RULES = (
    'REGRA DE PRECO (PIZZARIA):\n'
    '- O preco final de cada pizza e: preco base do sabor + modificador do tamanho + acrescimo da borda.\n'
    '- Sempre calcule e confirme o preco final antes de fechar o pedido.\n'
    '- Exemplo: "Calabresa Grande com borda de catupiry = R$55 + R$8 = R$63,00. Confirma?"'
)


def _normalize_objective(value: str | None) -> str:
    normalized = (value or '').strip().upper()
    if normalized == 'AGENDAR_HORARIO':
        return 'AGENDAR'
    if normalized in {'FECHAR_PEDIDO', 'AGENDAR', 'TIRAR_DUVIDAS'}:
        return normalized
    return 'FECHAR_PEDIDO'


def _detect_product(text: str) -> bool:
    normalized = (text or '').lower()
    return any(
        token in normalized
        for token in ('pizza', 'lanche', 'hamburg', 'combo', 'produto', 'pedido', 'quero')
    )


def _detect_size_or_quantity(text: str) -> bool:
    normalized = (text or '').lower()
    return bool(
        re.search(r'\b(\d+)\b', normalized)
        or any(token in normalized for token in ('p ', 'm ', 'g ', 'grande', 'medio', 'médio', 'pequena', 'pequeno', 'tamanho'))
    )


def _detect_payment(text: str) -> bool:
    normalized = (text or '').lower()
    return any(token in normalized for token in ('pix', 'cartao', 'cartão', 'dinheiro', 'dinheirinho', 'credito', 'crédito', 'debito', 'débito'))


def _detect_name(text: str) -> bool:
    normalized = (text or '').lower()
    if 'meu nome' in normalized or 'sou ' in normalized:
        return True
    return bool(re.search(r'\b[a-záàâãéêíóôõúç]{3,}\b', normalized, flags=re.IGNORECASE))


def _detect_datetime(text: str) -> bool:
    normalized = (text or '').lower()
    return bool(
        re.search(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', normalized)
        or re.search(r'\b\d{1,2}:\d{2}\b', normalized)
        or any(token in normalized for token in ('amanhã', 'amanha', 'hoje', 'segunda', 'terça', 'terca', 'quarta', 'quinta', 'sexta', 'sábado', 'sabado', 'domingo'))
    )


def _detect_service(text: str) -> bool:
    normalized = (text or '').lower()
    return any(token in normalized for token in ('serviço', 'servico', 'consulta', 'avaliação', 'avaliacao', 'limpeza', 'corte', 'sessão', 'sessao', 'agendar'))


def _infer_missing_checklist(history_window, user_message: str, objective: str, tenant_id: str | None = None) -> dict[str, bool]:
    active_messages = slice_messages_after_last_completed_order(list(history_window))
    human_texts = collect_active_human_texts(active_messages, include_user_message=user_message or '')
    corpus = ' '.join(human_texts)

    if objective == 'AGENDAR':
        return {
            'service': not _detect_service(corpus),
            'name': not _detect_name(corpus),
            'datetime': not _detect_datetime(corpus),
        }

    if objective == 'TIRAR_DUVIDAS':
        return {'closure': False}

    payload = build_order_payload_from_history_window(
        history_window=active_messages,
        user_message=user_message,
        tenant_id=tenant_id,
    )
    order_data = payload.get('order') or {}
    extracted_items = order_data.get('items') or []
    extracted_address = str(order_data.get('customer_address') or '').strip()
    extracted_payment = str(order_data.get('payment_method') or '').strip()
    if not extracted_items:
        confirmed_item = _extract_confirmed_item_from_ai(active_messages)
        if confirmed_item:
            qty = _extract_recent_quantity_from_humans(active_messages)
            extracted_items = [
                {
                    'product_name': confirmed_item,
                    'size': '',
                    'quantity': max(1, qty),
                }
            ]

    sub_nicho = str(order_data.get('sub_nicho') or '').strip().lower()
    size_missing = False
    if extracted_items and sub_nicho == 'pizzaria':
        size_missing = any(not str(item.get('size') or '').strip() for item in extracted_items)

    return {
        'product': not extracted_items,
        'size_or_quantity': size_missing or not _detect_size_or_quantity(corpus),
        'address': not extracted_address,
        'payment': not extracted_payment,
    }


def _next_question_from_missing(missing: dict[str, bool], objective: str) -> str:
    if objective == 'AGENDAR':
        if missing['service'] and missing['name']:
            return 'Para agendar agora: qual serviço você deseja e qual seu nome?'
        if missing['service']:
            return 'Qual serviço você deseja agendar?'
        if missing['name'] and missing['datetime']:
            return 'Perfeito. Qual seu nome e qual dia/horário você prefere?'
        if missing['name']:
            return 'Qual seu nome para eu confirmar o agendamento?'
        if missing['datetime']:
            return 'Qual dia e horário você prefere?'
        return ''

    if objective == 'TIRAR_DUVIDAS':
        return 'Ficou alguma dúvida?'

    if missing['product'] and missing['size_or_quantity']:
        return 'Qual produto você quer e qual tamanho/quantidade?'

    if missing['product']:
        return 'Qual produto você quer pedir?'

    if missing['size_or_quantity'] and missing['address']:
        return 'Para fechar rápido: qual tamanho/quantidade e o endereço completo (Rua, Número e Bairro)?'

    if missing['size_or_quantity'] and missing['payment']:
        return 'Para fechar agora: qual tamanho/quantidade e forma de pagamento?'

    if missing['size_or_quantity']:
        return 'Qual tamanho ou quantidade você quer?'

    if missing['address']:
        return 'Qual é o endereço completo para entrega (Rua, Número e Bairro)?'

    if missing['payment']:
        return 'Qual será a forma de pagamento (Pix, cartão ou dinheiro)?'

    return ''


def _contains_address_signal(text: str) -> bool:
    normalized = (text or '').lower()
    keywords = ('rua ', 'av ', 'avenida', 'bairro', 'número', 'numero', 'moro na', 'endereço', 'endereco')
    return any(k in normalized for k in keywords)


def _contains_order_signal(text: str) -> bool:
    normalized = (text or '').lower()
    keywords = ('quero', 'pedido', 'pizza', 'lanche', 'hamburg', 'combo', 'calabresa', 'frango')
    return any(k in normalized for k in keywords)


def _extract_payment_label(text: str) -> str:
    normalized = (text or '').lower()
    if 'pix' in normalized:
        return 'Pix'
    if any(token in normalized for token in ('cartao', 'cartão', 'credito', 'crédito', 'debito', 'débito')):
        return 'Cartão'
    if 'dinheiro' in normalized or 'dinheirinho' in normalized:
        return 'Dinheiro'
    return '[forma escolhida]'


def _extract_address_from_text(text: str) -> str:
    if not text:
        return ''

    normalized = text.strip()
    street_pattern = re.compile(
        r'(?i)(?:moro na\s+)?((?:rua|r\.|avenida|av\.?|travessa|tv\.?|alameda|estrada|rodovia)\s+.+)',
    )
    match = street_pattern.search(normalized)
    if not match:
        return ''

    address = match.group(1).strip(' ,.-')
    address = re.sub(
        r'(?i)(?:[\.,;:\- ]+)?(?:vou pagar|pagamento|pago no|pagar no|pix|cart[aã]o|dinheiro).*$','',
        address,
    ).strip(' ,.-')
    return address


def _extract_latest_address(history_window, user_message: str) -> str:
    human_texts = [str(msg.content).strip() for msg in history_window if msg.type == 'human']
    if user_message.strip():
        human_texts.append(user_message.strip())

    for text in reversed(human_texts):
        address = _extract_address_from_text(text)
        if address:
            return address

    return '[endereço informado pelo cliente]'


def _normalize_catalog_lookup(value: str) -> str:
    normalized = unicodedata.normalize('NFKD', value or '')
    normalized = ''.join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r'[^a-z0-9]+', ' ', normalized)
    return ' '.join(normalized.split())


def _extract_requested_size_label(text: str) -> str:
    normalized = _normalize_catalog_lookup(text)
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


def _extract_catalog_order_item(history_window, user_message: str, tenant_id: str | None) -> str:
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        return ''

    human_texts = [str(msg.content).strip() for msg in history_window if msg.type == 'human']
    if user_message.strip():
        human_texts.append(user_message.strip())

    product_lines = [text for text in human_texts if _contains_order_signal(text)]
    product_source = product_lines[-1] if product_lines else (user_message or '').strip()
    if not product_source:
        return ''

    try:
        stock_data = asyncio.run(fetch_stock_for_context(normalized_tenant_id))
    except Exception:
        return ''

    normalized_source = _normalize_catalog_lookup(product_source)
    if not normalized_source:
        return ''

    if str(stock_data.get('sub_nicho') or '').strip().lower() == 'pizzaria':
        candidates = [
            str(sabor.get('nome') or '').strip()
            for sabor in stock_data.get('sabores', [])
            if str(sabor.get('nome') or '').strip()
        ]
    else:
        candidates = [
            str(item.get('nome') or '').strip()
            for item in stock_data.get('items', [])
            if str(item.get('nome') or '').strip()
        ]

    best_match = ''
    best_length = -1
    for candidate in candidates:
        normalized_candidate = _normalize_catalog_lookup(candidate)
        if not normalized_candidate:
            continue
        if normalized_candidate in normalized_source and len(normalized_candidate) > best_length:
            best_match = candidate
            best_length = len(normalized_candidate)

    if not best_match:
        return ''

    quantity_match = re.search(r'\b(\d+)\s*[xX]?\b', normalized_source)
    quantity = int(quantity_match.group(1)) if quantity_match else 1
    size_label = _extract_requested_size_label(product_source)

    item_name = best_match
    if size_label:
        item_name = f'{item_name} ({size_label})'

    return f'{max(1, quantity)}x {item_name}'


def _extract_order_context_from_history(history_window, user_message: str, tenant_id: str | None = None) -> tuple[str, str, str]:
    payload = build_order_payload_from_history_window(
        history_window=history_window,
        user_message=user_message,
        tenant_id=tenant_id,
    )
    order_data = payload.get('order') or {}
    items = str(order_data.get('items_text') or '[itens informados pelo cliente]').strip()
    address = str(order_data.get('customer_address') or '[endereço informado pelo cliente]').strip()
    payment = str(order_data.get('payment_method') or '[forma escolhida]').strip()

    if _is_placeholder_value(items):
        confirmed_item = _extract_confirmed_item_from_ai(history_window)
        if confirmed_item:
            if re.search(r'^\s*\d+\s*[xX]?\s+', confirmed_item):
                items = confirmed_item
            else:
                qty = _extract_recent_quantity_from_humans(history_window)
                items = f'{max(1, qty)}x {confirmed_item}'

    if _is_placeholder_value(payment):
        extracted_payment = _extract_payment_label(user_message)
        if extracted_payment != '[forma escolhida]':
            payment = extracted_payment

    return items, address, payment


def _extract_summary_field(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ''
    return (match.group(1) or '').strip(' \n\t*-_')


def _format_brl(value: float) -> str:
    return f'R$ {value:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def _estimate_total_from_items(items_text: str, tenant_id: str | None = None) -> float:
    normalized_tenant_id = (tenant_id or '').strip()
    if not normalized_tenant_id:
        return 0.0

    try:
        stock_data = asyncio.run(fetch_stock_for_context(normalized_tenant_id))
    except Exception:
        return 0.0

    price_map: dict[str, float] = {}
    for item in stock_data.get('items', []):
        name = _normalize_catalog_lookup(str(item.get('nome') or '').strip())
        if not name:
            continue
        try:
            price = float(item.get('preco') or 0)
        except (TypeError, ValueError):
            continue
        if price > 0:
            price_map[name] = price

    total = 0.0
    parts = [part.strip() for part in str(items_text or '').split('+') if part.strip()]
    for part in parts:
        quantity = 1
        product_text = part
        match = re.match(r'^\s*(\d+)\s*(?:[xX]\s*)?(.+)$', part)
        if match:
            quantity = max(1, int(match.group(1)))
            product_text = match.group(2).strip()

        normalized_name = _normalize_catalog_lookup(re.sub(r'\s*\(.*\)$', '', product_text))
        if not normalized_name:
            continue

        resolved = 0.0
        if normalized_name in price_map:
            resolved = price_map[normalized_name]
        else:
            best = ''
            for candidate in price_map.keys():
                if candidate in normalized_name or normalized_name in candidate:
                    if len(candidate) > len(best):
                        best = candidate
            if best:
                resolved = price_map[best]

        total += resolved * quantity

    return total


def _format_order_final_summary(items: str, address: str, payment: str, tenant_id: str | None = None) -> str:
    safe_items = items or '[lista detalhada com tamanhos]'
    safe_address = address or '[endereço completo do cliente]'
    safe_payment = payment or '[forma escolhida]'
    total = _estimate_total_from_items(safe_items, tenant_id)
    total_line = f'💰 *Total: {_format_brl(total)}*\n' if total > 0 else ''

    return (
        '✅ *Pedido anotado!*\n\n'
        f'🛒 *Itens:* {safe_items}\n'
        f'{total_line}'
        f'📍 *Endereço:* {safe_address}\n'
        f'💳 *Pagamento:* {safe_payment}\n'
        '\nTô separando já! Em breve saiu 🚀\n'
        '✅'
    )


def _normalize_order_final_summary(text: str, tenant_id: str | None = None) -> str:
    items = _extract_summary_field(
        text,
        r'(?:🛒\s*\*?itens:?\*?)\s*(.+?)(?=(?:📍|\*?endere|💳|\*?pagamento|_?obrigado|$))',
    )
    address = _extract_summary_field(
        text,
        r'(?:📍\s*\*?endere[cç]o:?\*?)\s*(.+?)(?=(?:💳|\*?pagamento|_?obrigado|$))',
    )
    payment = _extract_summary_field(
        text,
        r'(?:💳\s*\*?pagamento:?\*?)\s*(.+?)(?=(?:_?obrigado|$))',
    )
    return _format_order_final_summary(items, address, payment, tenant_id)


def _format_schedule_final_summary(service: str, customer_name: str, datetime_slot: str) -> str:
    safe_service = service or '[Serviço escolhido]'
    safe_name = customer_name or '[Nome]'
    safe_datetime = datetime_slot or '[Dia e Hora]'
    return (
        '*Agendamento Confirmado!* ✅\n'
        f'👤 *Cliente:* {safe_name}\n'
        f'📅 *Data/Hora:* {safe_datetime}\n'
        f'✨ *Serviço:* {safe_service}\n'
        '_Obrigado! Te esperamos no horário marcado._'
    )


def _normalize_schedule_final_summary(text: str) -> str:
    customer_name = _extract_summary_field(
        text,
        r'(?:👤\s*\*?cliente:?\*?)\s*(.+?)(?=(?:📅|✨|_?obrigado|$))',
    )
    datetime_slot = _extract_summary_field(
        text,
        r'(?:📅\s*\*?data/hora:?\*?)\s*(.+?)(?=(?:✨|_?obrigado|$))',
    )
    service = _extract_summary_field(
        text,
        r'(?:✨\s*\*?servi[cç]o:?\*?)\s*(.+?)(?=(?:_?obrigado|$))',
    )
    return _format_schedule_final_summary(service, customer_name, datetime_slot)


def _format_faq_final_summary() -> str:
    return (
        '*Atendimento Encerrado* ✅\n'
        '_Espero ter ajudado! Qualquer nova dúvida, é só chamar._'
    )


def _is_final_message(text: str, objective: str) -> bool:
    if objective == 'FECHAR_PEDIDO':
        return (text or '').rstrip().endswith('✅')

    marker = _FINAL_MARKERS.get(objective, 'pedido confirmado')
    return marker in (text or '').lower()


def _should_close_faq(user_message: str) -> bool:
    normalized = (user_message or '').lower()
    return any(token in normalized for token in ('obrigado', 'obrigada', 'só isso', 'so isso', 'encerra', 'encerrar', 'é isso', 'e isso'))


def _sanitize_persona_response(text: str, objective: str, tenant_id: str | None = None) -> str:
    """Aplica guardrails finais para manter o estilo de WhatsApp enxuto."""
    cleaned = (text or '').strip()
    if not cleaned:
        return cleaned

    # Remove links externos por segurança.
    cleaned = _URL_PATTERN.sub('', cleaned)

    # Evita início robótico/burocrático.
    cleaned = _LEADING_ROBOTIC_WORDS.sub('', cleaned).strip()

    # Se já for final, normaliza no template absoluto do módulo.
    if _is_final_message(cleaned, objective):
        if objective == 'AGENDAR':
            return _normalize_schedule_final_summary(cleaned)
        if objective == 'TIRAR_DUVIDAS':
            return _format_faq_final_summary()
        return _normalize_order_final_summary(cleaned, tenant_id)

    # Corta mensagens muito longas para no máximo 3 frases curtas.
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+|\n+', cleaned) if p.strip()]
    if len(parts) > 3:
        cleaned = ' '.join(parts[:3]).strip()

    return cleaned


def _enforce_sales_funnel(
    model_reply: str,
    history_window,
    user_message: str,
    objective: str,
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> str:
    cleaned = _sanitize_persona_response(model_reply, objective, tenant_id)
    should_mark_confirmed = False
    produto_confirmado_em_sessao = False
    phone = ''
    if objective == 'FECHAR_PEDIDO' and tenant_id and session_id and ':' in session_id:
        _, phone = session_id.split(':', 1)
        phone = phone.strip()
        if phone:
            try:
                redis_client = _get_redis_sync()
                if redis_client is not None:
                    produto_confirmado_em_sessao = bool(
                        redis_client.get(f'session:{tenant_id}:{phone}:produto_confirmado')
                    )
            except Exception:
                produto_confirmado_em_sessao = False

    missing = _infer_missing_checklist(history_window, user_message, objective, tenant_id)
    payload = build_order_payload_from_history_window(
        history_window=history_window,
        user_message=user_message,
        tenant_id=tenant_id,
    )
    if objective == 'FECHAR_PEDIDO' and tenant_id and session_id:
        should_mark_confirmed = bool((payload.get('analytics') or {}).get('produto_confirmado'))
        if not should_mark_confirmed:
            should_mark_confirmed = (
                _is_affirmative_text(user_message)
                and _recent_ai_requested_product_confirmation(history_window)
            )

        if phone and should_mark_confirmed:
            try:
                redis_client = _get_redis_sync()
                if redis_client is not None:
                    redis_client.setex(
                        f'session:{tenant_id}:{phone}:produto_confirmado',
                        30 * 60,
                        '1',
                    )
            except Exception:
                pass

    if objective == 'FECHAR_PEDIDO' and (produto_confirmado_em_sessao or should_mark_confirmed):
        lowered = cleaned.lower()
        if (
            'você quis dizer' in lowered
            or 'voce quis dizer' in lowered
            or 'para confirmar certinho' in lowered
        ):
            items, address, payment = _extract_order_context_from_history(
                history_window,
                user_message,
                tenant_id,
            )
            if not address:
                return random.choice([
                    'Show! Anotado 🍺 Me manda o endereço?',
                    'Fechado! Me passa o endereço pra entrega?',
                    'Boa! Vou separar já. Endereço?',
                    'Perfeito! Entrega ou retirada?',
                ])
            if not payment:
                return 'Perfeito, endereço anotado. Qual a forma de pagamento?'
            return _format_order_final_summary(items, address, payment, tenant_id)

    if objective == 'FECHAR_PEDIDO' and _is_order_closure_text(user_message):
        items, address, payment = _extract_order_context_from_history(
            history_window,
            user_message,
            tenant_id,
        )
        if _is_placeholder_value(items):
            confirmed_item = _extract_confirmed_item_from_ai(history_window)
            if confirmed_item:
                quantity = _extract_quantity_from_text(user_message)
                if quantity <= 1:
                    quantity = _extract_recent_quantity_from_humans(history_window)
                if re.search(r'^\s*\d+\s*[xX]?\s+', confirmed_item):
                    items = confirmed_item
                else:
                    items = f'{max(1, quantity)}x {confirmed_item}'

        if not payment:
            extracted_payment = _extract_payment_label(user_message)
            if extracted_payment != '[forma escolhida]':
                payment = extracted_payment

        if _is_placeholder_value(address):
            address = ''
        if _is_placeholder_value(payment):
            payment = ''

        if not address:
            if payment and payment != '[forma escolhida]':
                return f'Anotado! Pagamento no {payment}. Me manda o endereço pra entrega 📍'
            return 'Ótimo! Me manda o endereço pra entrega 📍'

        if not payment or payment == '[forma escolhida]':
            return 'Qual forma de pagamento? Pix, dinheiro ou cartão?'

        if _is_placeholder_value(items):
            return 'Perfeito! Só me confirma o item e a quantidade para fechar.'

        return _format_order_final_summary(items, address, payment, tenant_id)
    order_data = payload.get('order') or {}

    if objective == 'TIRAR_DUVIDAS' and _should_close_faq(user_message):
        return _format_faq_final_summary()

    if objective == 'TIRAR_DUVIDAS':
        if 'dúvida' not in cleaned.lower() and 'duvida' not in cleaned.lower():
            cleaned = f'{cleaned} Ficou alguma dúvida?'.strip()

    if _is_final_message(cleaned, objective) and not any(missing.values()):
        return cleaned

    if _is_final_message(cleaned, objective) and any(missing.values()):
        cleaned = ''

    if objective == 'FECHAR_PEDIDO' and not any(missing.values()):
        items, address, payment = _extract_order_context_from_history(
            history_window,
            user_message,
            tenant_id,
        )
        return _format_order_final_summary(items, address, payment, tenant_id)

    if objective == 'FECHAR_PEDIDO' and missing.get('product'):
        suggested_name = str(order_data.get('suggested_product_name') or '').strip()
        suggested_score = float(order_data.get('suggested_product_score') or 0)
        if produto_confirmado_em_sessao or should_mark_confirmed:
            items, address, payment = _extract_order_context_from_history(
                history_window,
                user_message,
                tenant_id,
            )
            confirmed_qty = _extract_quantity_from_text(user_message)
            extracted_item = _extract_catalog_order_item(history_window, user_message, tenant_id)
            if extracted_item:
                items = extracted_item
            elif not items or '[' in items:
                confirmed_item = _extract_confirmed_item_from_ai(history_window)
                if confirmed_item:
                    if re.search(r'^\s*\d+\s*[xX]?\s+', confirmed_item):
                        items = confirmed_item
                    else:
                        items = f'{confirmed_qty}x {confirmed_item}'
                elif suggested_name:
                    qty = confirmed_qty if confirmed_qty > 0 else 1
                    items = f'{qty}x {suggested_name}'

            if not payment:
                extracted_payment = _extract_payment_label(user_message)
                if extracted_payment != '[forma escolhida]':
                    payment = extracted_payment

            if _is_placeholder_value(address):
                address = ''
            if _is_placeholder_value(payment):
                payment = ''

            if _is_placeholder_value(items):
                return 'Perfeito! Me confirma rapidinho o produto para eu fechar certinho.'

            if not address:
                return random.choice([
                    'Show! Anotado 🍺 Me manda o endereço?',
                    'Fechado! Me passa o endereço pra entrega?',
                    'Boa! Vou separar já. Endereço?',
                    'Perfeito! Entrega ou retirada?',
                ])
            if not payment:
                return 'Perfeito, endereço anotado. Qual a forma de pagamento?'
            if _is_placeholder_value(items):
                return 'Perfeito! Só me confirma o item e a quantidade para eu fechar.'
            return _format_order_final_summary(items, address, payment, tenant_id)

        if suggested_name and suggested_score >= 0.45:
            return random.choice([
                f'{suggested_name}, certo? 🍺',
                f'{suggested_name}, pode ser?',
                f'É {suggested_name} mesmo?',
                f'{suggested_name} então, tô anotando - confirma?',
            ])

    next_question = _next_question_from_missing(missing, objective)
    if not next_question:
        return cleaned

    if '?' in cleaned:
        return cleaned

    return f'{cleaned} {next_question}'.strip()


def _openai_key_is_configured() -> bool:
    key = (OPENAI_API_KEY or '').strip()
    return bool(key and key.upper() != 'YOUR_KEY')


def _provider_unavailable_fallback(
    objective: str,
    history_window,
    user_message: str,
    tenant_id: str | None = None,
) -> str:
    if objective == 'AGENDAR':
        missing = _infer_missing_checklist(history_window, user_message, objective, tenant_id)
        next_question = _next_question_from_missing(missing, objective)
        return next_question or 'Perfeito. Vou confirmar seu agendamento.'

    if objective == 'TIRAR_DUVIDAS':
        return 'Pode me dizer sua dúvida em uma frase?'

    missing = _infer_missing_checklist(history_window, user_message, objective, tenant_id)
    next_question = _next_question_from_missing(missing, objective)
    if next_question:
        return next_question

    return _format_order_final_summary(
        items='[itens informados pelo cliente]',
        address='[endereço informado pelo cliente]',
        payment='[pagamento informado pelo cliente]',
        tenant_id=tenant_id,
    )


def generate_persona_response(
    instruction: str,
    user_message: str,
    session_id: str,
    persona_system_prompt: str | None = None,
    objective_mode: str | None = None,
    tenant_id: str | None = None,
) -> str:
    """Generates a persona-styled response using Gemini official SDK."""
    history = get_session_history(session_id)
    objective = _normalize_objective(objective_mode)
    base_prompt = (persona_system_prompt or '').strip() or _PERSONA_SYSTEM_PROMPT
    module_rules = _FUNNEL_MODULES_RULES.get(objective, _FUNNEL_MODULES_RULES['FECHAR_PEDIDO'])

    dynamic_hint = ''
    if objective == 'FECHAR_PEDIDO' and _contains_order_signal(user_message) and _contains_address_signal(user_message):
        dynamic_hint = (
            'HINT DE FECHAMENTO RÁPIDO: O cliente já informou pedido e endereço nesta mensagem. '
            'NÃO repita elogios. Pergunte imediatamente os dados faltantes para fechar, '
            'priorizando tamanho/quantidade e forma de pagamento na mesma resposta.'
        )

    cardapio_context = 'Catálogo indisponível no momento.'
    normalized_tenant_id = (tenant_id or '').strip()
    sub_nicho = None
    tenant_runtime_config: dict = {}
    if normalized_tenant_id:
        try:
            tenant_runtime_config = asyncio.run(get_tenant_configs(normalized_tenant_id))
        except Exception as error:
            print(f'[TENANT-CONFIG] Falha ao carregar config para prompt: {error}')

    if normalized_tenant_id:
        try:
            sub_nicho = asyncio.run(get_tenant_sub_nicho(normalized_tenant_id))
        except Exception as error:
            print(f'[SUB-NICHO] Falha ao carregar sub_nicho para prompt: {error}')

        # Para pizzaria, o catálogo vem inteiramente do stock_block (stock_items + pizza_tamanhos/bordas).
        # Chamar get_cardapio_context retornaria "nenhum produto" (tabela produtos vazia) e
        # confundiria o LLM, que ignoraria o stock_block correto.
        if sub_nicho != 'pizzaria':
            try:
                cardapio_context = asyncio.run(get_cardapio_context(normalized_tenant_id))
            except Exception as error:
                cardapio_context = f'Falha ao carregar catálogo do tenant: {error}'

    # Injetar contexto de estoque/cardapio por sub-nicho
    stock_block = ''
    niche_block = ''
    if sub_nicho:
        niche_block = get_nicho_prompt(sub_nicho, tenant_runtime_config)
    if not niche_block and sub_nicho in _TEST_DRIVE_SUB_NICHO_RULES:
        niche_block = _TEST_DRIVE_SUB_NICHO_RULES[sub_nicho]

    phone_for_context = ''
    if ':' in session_id:
        _, phone_for_context = session_id.split(':', 1)
        phone_for_context = phone_for_context.strip()
    dynamic_session_context = ''
    if normalized_tenant_id and phone_for_context:
        try:
            dynamic_session_context = asyncio.run(
                get_dynamic_context(
                    normalized_tenant_id,
                    phone_for_context,
                    _get_redis_async(),
                )
            )
        except Exception:
            dynamic_session_context = ''

    if normalized_tenant_id and sub_nicho in {'adega', 'pizzaria'}:
        try:
            stock_context = get_stock_context(normalized_tenant_id)
            extra_rules = _ADEGA_STOCK_RULES if sub_nicho == 'adega' else _PIZZARIA_PRICE_RULES
            stock_block = (
                f'\n\n[{stock_context}]\n\n'
                f'{extra_rules}'
            )
            # Para pizzaria, o stock_block É o catálogo — substituir o placeholder
            if sub_nicho == 'pizzaria':
                cardapio_context = ''  # evita "Catálogo indisponível" no prompt
        except Exception as error:
            print(f'[ESTOQUE-CONTEXT] Falha ao carregar estoque para prompt: {error}')

    effective_system_prompt = (
        f'{base_prompt}\n\n'
        f'{niche_block}\n\n'
        f'{dynamic_session_context}\n'
        f'{_WHATSAPP_HARD_RULES}\n\n'
        f'{_CATALOGO_GOLD_RULE}\n\n'
        f'{_ALTERNATIVE_WHEN_UNAVAILABLE_RULE}\n\n'
        f'[CATÁLOGO DA LOJA E REGRAS]\n'
        f'{cardapio_context}'
        f'{stock_block}\n\n'
        f'{module_rules}'
    )
    if dynamic_hint:
        effective_system_prompt = f'{effective_system_prompt}\n\n{dynamic_hint}'

    if sub_nicho == 'adega':
        effective_system_prompt = f'{effective_system_prompt}\n\n{_ADEGA_CONTEXT_CRITICAL_RULE}'

    history_window = slice_messages_after_last_completed_order(list(history.messages))[-_HISTORY_WINDOW_SIZE:]

    history_lines: list[str] = []
    for msg in history_window:
        role = 'Usuário' if msg.type == 'human' else 'Assistente'
        history_lines.append(f'{role}: {msg.content}')

    content = ''
    gemini_error: str | None = None

    if GEMINI_API_KEY and GEMINI_API_KEY.strip():
        try:
            client = genai.Client(api_key=GEMINI_API_KEY.strip())
            gemini_user_prompt = (
                f'Histórico recente:\n{chr(10).join(history_lines) if history_lines else "(sem histórico)"}\n\n'
                f'Mensagem atual do usuário: {user_message}\n\n'
                'Use o histórico acima como fonte de verdade para manter continuidade entre turnos. '
                'Responda em português do Brasil.'
            )

            configured_model = (GEMINI_MODEL_NAME or '').strip()
            fast_model = (os.getenv('GEMINI_FAST_MODEL') or 'gemini-2.5-flash').strip()
            model_candidates = [m for m in [fast_model, configured_model, 'gemini-1.5-flash'] if m]
            # remove duplicatas preservando ordem
            model_candidates = list(dict.fromkeys(model_candidates))

            response = None
            last_error: Exception | None = None
            for candidate_model in model_candidates:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    client.models.generate_content,
                    model=candidate_model,
                    contents=gemini_user_prompt,
                    config=types.GenerateContentConfig(
                        temperature=OPENAI_MODEL_TEMPERATURE,
                        system_instruction=(
                            f'{effective_system_prompt}\n\n'
                            f'INSTRUÇÃO DO SISTEMA: {instruction}'
                        ),
                    ),
                )
                try:
                    response = future.result(timeout=8.0)
                    logger.info('[GEMINI] Modelo usado: %s', candidate_model)
                    break
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    logger.warning('[GEMINI] Timeout com modelo %s', candidate_model)
                    last_error = None
                except Exception as model_error:
                    last_error = model_error
                    logger.warning('[GEMINI] Falha com modelo %s: %s', candidate_model, model_error)
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

            if response is None:
                if last_error is not None:
                    raise last_error
                logger.error('[GEMINI] Timeout na chamada — retornando fallback')
                fallback = (
                    'Desculpa a demora! Pode continuar com seu pedido? 😊 '
                    'Me fala se quer mais alguma coisa ou pode fechar!'
                )
                history.add_user_message(user_message)
                history.add_ai_message(fallback)
                return fallback

            content = (getattr(response, 'text', None) or '').strip()
        except Exception as e:
            logger.error(f'[GEMINI] Erro: {e}')
            print("\n" + "=" * 50)
            print(f"🔥 ERRO FATAL NA CHAMADA DA IA (GEMINI): {str(e)}")
            print("=" * 50)
            traceback.print_exc()
            print("=" * 50 + "\n")
            fallback = 'Desculpa, tive um problema técnico. Pode repetir o pedido?'
            history.add_user_message(user_message)
            history.add_ai_message(fallback)
            return fallback

    if not content:
        reasons: list[str] = []
        if gemini_error:
            reasons.append(f'Gemini falhou: {gemini_error}')
        if not GEMINI_API_KEY:
            reasons.append('Nenhuma chave de IA válida configurada (GEMINI_API_KEY).')
        # Mantém o atendimento vivo mesmo com indisponibilidade temporária de provedores.
        # O motivo pode ser útil para troubleshooting via logs.
        reason_text = ' | '.join(reasons) if reasons else 'Falha desconhecida ao gerar conteúdo.'
        print(f'[LLM-FALLBACK] {reason_text}')
        content = _provider_unavailable_fallback(
            objective,
            history_window,
            user_message,
            normalized_tenant_id,
        )

    content = _enforce_sales_funnel(
        content,
        history_window,
        user_message,
        objective,
        normalized_tenant_id,
        session_id,
    )

    history.add_user_message(user_message)
    history.add_ai_message(content)
    return content


def invoke_rag_chain(user_message: str, session_id: str) -> str:
    """Invokes the conversational RAG chain (saves history automatically)."""
    chain = get_conversational_rag_chain()
    result = chain.invoke(
        {'input': user_message},
        config={'configurable': {'session_id': session_id}},
    )
    return result.get('answer', '')
