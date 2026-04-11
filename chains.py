import re
import json
import asyncio
import traceback

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from google import genai
from google.genai import types

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    OPENAI_MODEL_TEMPERATURE,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
)
from database_api import fetch_active_produtos, get_tenant_configs
from memory import get_session_history
from vectorstore import get_vectorstore
from prompts import contextualize_prompt, qa_prompt


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
    'Você é um assistente comercial de WhatsApp. '
    'Seja natural, pragmático e orientado à conversão com o menor número de mensagens possível.'
)

_CATALOGO_GOLD_RULE = (
    'Atenção: Você só pode oferecer e vender os produtos que estão listados no CATÁLOGO abaixo. '
    'Respeite rigorosamente as regras de Variações, Extras e Limites descritas no contexto de cada item. '
    'Não invente produtos ou preços.'
)

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
    return any(token in normalized for token in ('pix', 'cartao', 'cartão', 'dinheiro', 'credito', 'crédito', 'debito', 'débito'))


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


def _infer_missing_checklist(history_window, user_message: str, objective: str) -> dict[str, bool]:
    human_texts = [str(msg.content) for msg in history_window if msg.type == 'human']
    human_texts.append(user_message or '')
    corpus = ' '.join(human_texts)

    if objective == 'AGENDAR':
        return {
            'service': not _detect_service(corpus),
            'name': not _detect_name(corpus),
            'datetime': not _detect_datetime(corpus),
        }

    if objective == 'TIRAR_DUVIDAS':
        return {'closure': False}

    return {
        'product': not _detect_product(corpus),
        'size_or_quantity': not _detect_size_or_quantity(corpus),
        'address': not _contains_address_signal(corpus),
        'payment': not _detect_payment(corpus),
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
    if 'dinheiro' in normalized:
        return 'Dinheiro'
    return '[forma escolhida]'


def _extract_order_context_from_history(history_window, user_message: str) -> tuple[str, str, str]:
    human_texts = [str(msg.content).strip() for msg in history_window if msg.type == 'human']
    if user_message.strip():
        human_texts.append(user_message.strip())

    product_lines = [text for text in human_texts if _contains_order_signal(text)]
    address_lines = [text for text in human_texts if _contains_address_signal(text)]
    corpus = ' '.join(human_texts)

    items = product_lines[-1] if product_lines else '[itens informados pelo cliente]'
    address = address_lines[-1] if address_lines else '[endereço informado pelo cliente]'
    payment = _extract_payment_label(corpus)
    return items, address, payment


def _extract_summary_field(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ''
    return (match.group(1) or '').strip(' \n\t*-_')


def _format_order_final_summary(items: str, address: str, payment: str) -> str:
    safe_items = items or '[lista detalhada com tamanhos]'
    safe_address = address or '[endereço completo do cliente]'
    safe_payment = payment or '[forma escolhida]'

    return (
        '*Resumo do Pedido*\n'
        f'🛒 *Itens:* {safe_items}\n'
        f'📍 *Endereço:* {safe_address}\n'
        f'💳 *Pagamento:* {safe_payment}\n'
        '✅'
    )


def _normalize_order_final_summary(text: str) -> str:
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
    return _format_order_final_summary(items, address, payment)


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


def _sanitize_persona_response(text: str, objective: str) -> str:
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
        return _normalize_order_final_summary(cleaned)

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
) -> str:
    cleaned = _sanitize_persona_response(model_reply, objective)

    missing = _infer_missing_checklist(history_window, user_message, objective)

    if objective == 'TIRAR_DUVIDAS' and _should_close_faq(user_message):
        return _format_faq_final_summary()

    if objective == 'TIRAR_DUVIDAS':
        if 'dúvida' not in cleaned.lower() and 'duvida' not in cleaned.lower():
            cleaned = f'{cleaned} Ficou alguma dúvida?'.strip()

    if _is_final_message(cleaned, objective):
        return cleaned

    if objective == 'FECHAR_PEDIDO' and not any(missing.values()):
        items, address, payment = _extract_order_context_from_history(
            history_window,
            user_message,
        )
        return _format_order_final_summary(items, address, payment)

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
) -> str:
    if objective == 'AGENDAR':
        missing = _infer_missing_checklist(history_window, user_message, objective)
        next_question = _next_question_from_missing(missing, objective)
        return next_question or 'Perfeito. Vou confirmar seu agendamento.'

    if objective == 'TIRAR_DUVIDAS':
        return 'Pode me dizer sua dúvida em uma frase?'

    missing = _infer_missing_checklist(history_window, user_message, objective)
    next_question = _next_question_from_missing(missing, objective)
    if next_question:
        return next_question

    return _format_order_final_summary(
        items='[itens informados pelo cliente]',
        address='[endereço informado pelo cliente]',
        payment='[pagamento informado pelo cliente]',
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
    if normalized_tenant_id:
        try:
            cardapio_context = asyncio.run(get_cardapio_context(normalized_tenant_id))
        except Exception as error:
            cardapio_context = f'Falha ao carregar catálogo do tenant: {error}'

    effective_system_prompt = (
        f'{base_prompt}\n\n'
        f'{_WHATSAPP_HARD_RULES}\n\n'
        f'{_CATALOGO_GOLD_RULE}\n\n'
        f'[CATÁLOGO DA LOJA E REGRAS]\n'
        f'{cardapio_context}\n\n'
        f'{module_rules}'
    )
    if dynamic_hint:
        effective_system_prompt = f'{effective_system_prompt}\n\n{dynamic_hint}'

    history_window = list(history.messages)[-_HISTORY_WINDOW_SIZE:]

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
                'Responda em português do Brasil.'
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=gemini_user_prompt,
                config=types.GenerateContentConfig(
                    temperature=OPENAI_MODEL_TEMPERATURE,
                    system_instruction=(
                        f'{effective_system_prompt}\n\n'
                        f'INSTRUÇÃO DO SISTEMA: {instruction}'
                    ),
                ),
            )

            content = (getattr(response, 'text', None) or '').strip()
        except Exception as e:
            print("\n" + "=" * 50)
            print(f"🔥 ERRO FATAL NA CHAMADA DA IA (GEMINI): {str(e)}")
            print("=" * 50)
            traceback.print_exc()
            print("=" * 50 + "\n")
            gemini_error = str(e)

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
        )

    content = _enforce_sales_funnel(
        content,
        history_window,
        user_message,
        objective,
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
