from __future__ import annotations

import re
import logging
from enum import Enum
from typing import Any
from difflib import SequenceMatcher
import unicodedata

try:
    from rapidfuzz.fuzz import token_set_ratio  # pyright: ignore[reportMissingImports]
except Exception:
    def token_set_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100


logger = logging.getLogger(__name__)


class PizzaState(Enum):
    AGUARDANDO_PEDIDO = 'aguardando_pedido'
    CONFIRMANDO_SABOR = 'confirmando_sabor'
    AGUARDANDO_BORDA = 'aguardando_borda'
    AGUARDANDO_ENDERECO = 'aguardando_endereco'
    AGUARDANDO_PAGAMENTO = 'aguardando_pagamento'
    FINALIZADO = 'finalizado'


TAMANHO_PATTERNS: dict[str, list[str]] = {
    'P': [r'\bP\b', r'\bpequen[ao]\b', r'\bpequena\b'],
    'M': [r'\bM\b', r'\bm[ée]di[ao]\b', r'\bmedia\b'],
    'G': [r'\bG\b', r'\bgrande\b'],
    'GG': [r'\bGG\b', r'\bfamili[ao]r\b', r'\bextra grande\b'],
}

PAYMENT_PATTERNS = [
    r'\b(pix|dinheiro|cart[aã]o|d[eé]bito|cr[eé]dito|dinheirinho)\b',
]

ADDRESS_PATTERNS = [
    r'(rua|av\.?|avenida|alameda|travessa|estrada|rodovia).{5,}',
    r'(moro|fico|entrega).{5,}',
    r'(bairro|n[uú]mero|n°|nº).{3,}',
]

AFFIRMATIVE = [
    r'^(sim|s|isso|pode ser|certo|exato|confirmo|isso mesmo|pode|t[aá] bom|ok|yes)[\s!.]*$',
    # Confirmações com contexto
    r'\b(confirmo|isso mesmo|pode ser|pode isso)\b',
    # Confirmações agressivas
    r'(j[aá] falei|falei que sim|[eé] sim|claro que sim|[oó]bvio)',
    # Repetições frustradas
    r'(sim[\s,!]+sim|sim{2,})',
]

NEGATIVE = [
    r'^(n[aã]o|nao|n|negativo|errado|outro|diferente)[\s!.]*$',
]

CLOSE_ORDER = [
    r'(s[oó] isso|[eé] isso|pode fechar|fecha|por enquanto [eé] isso|t[aá] bom assim|mais n[aã]o)',
]

TAMANHO_LABEL = {
    'P': 'Pequena',
    'M': 'Media',
    'G': 'Grande',
    'GG': 'GG',
}


def _normalize_text(value: str) -> str:
    return (value or '').strip()


def _normalize_for_match(value: str) -> str:
    raw = unicodedata.normalize('NFKD', value or '')
    without_accents = ''.join(ch for ch in raw if not unicodedata.combining(ch))
    lowered = without_accents.lower()
    return re.sub(r'[^a-z0-9\s]+', ' ', lowered).strip()


def _extract_with_patterns(text: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def extract_tamanho(text: str) -> tuple[str, str] | None:
    for tamanho in ('GG', 'G', 'M', 'P'):
        for pattern in TAMANHO_PATTERNS[tamanho]:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            raw = (match.group(0) or '').strip()
            if raw and raw.upper() in {'P', 'M', 'G', 'GG'}:
                return tamanho, raw.upper()
            return tamanho, TAMANHO_LABEL.get(tamanho, tamanho)
    return None


def extract_sabor(text: str, cardapio: list[dict[str, Any]]) -> tuple[str, float] | None:
    logger.info(f"[PIZZA] extract_sabor input: '{text}'")
    logger.info(f"[PIZZA] cardapio disponível: {[str(i.get('nome') or '').strip() for i in cardapio]}")

    text_lower = _normalize_for_match(text).strip()
    if not text_lower:
        logger.info('[PIZZA] melhor match: None score: 0.0')
        return None

    stopwords = [
        r'^opa[,\s]+',
        r'^quero\s+',
        r'^me\s+ve\s+',
        r'^me\s+da\s+',
        r'^uma\s+pizza\s+de\s+',
        r'^uma\s+pizza\s+',
        r'^pizza\s+de\s+',
        r'^pizza\s+',
        r'\bpor\s+favor\b',
        r'^pf[,\s]+',
        r'^pfv[,\s]+',
    ]

    text_clean = text_lower
    for sw in stopwords:
        text_clean = re.sub(sw, '', text_clean, flags=re.IGNORECASE).strip()

    logger.info(f"[PIZZA] texto limpo para match: '{text_clean}'")

    best_match = None
    best_score = 0.0

    for item in cardapio:
        nome = str(item.get('nome') or '').strip()
        if not nome:
            continue

        nome_norm = _normalize_for_match(nome)
        score_original = token_set_ratio(text_lower, nome_norm) / 100
        score_limpo = token_set_ratio(text_clean or text_lower, nome_norm) / 100
        score = max(score_original, score_limpo)

        logger.info(
            f"[PIZZA] '{nome}' -> original:{score_original:.2f} limpo:{score_limpo:.2f}"
        )

        if score > best_score:
            best_score = score
            best_match = nome

    logger.info(f"[PIZZA] melhor match: {best_match} score: {best_score}")

    if best_match and best_score >= 0.45:
        return best_match, best_score
    return None


def extract_borda(text: str, bordas: list[dict[str, Any]]) -> str | None:
    text_lower = text.lower()
    if re.search(r'(sem borda|n[aã]o quero borda|nao quero borda|normal)', text_lower):
        return 'Sem borda'

    best_match = None
    best_score = 0.0
    for borda in bordas:
        nome = str(borda.get('nome') or '').strip()
        if not nome or nome.lower() == 'sem borda':
            continue
        score = token_set_ratio(text_lower, nome.lower()) / 100
        if score > best_score:
            best_score = score
            best_match = nome

    if best_match and best_score >= 0.5:
        return best_match
    return None


def extract_payment(text: str) -> str | None:
    normalize = {
        'pix': 'Pix',
        'dinheiro': 'Dinheiro',
        'dinheirinho': 'Dinheiro',
        'cartao': 'Cartão',
        'cartão': 'Cartão',
        'débito': 'Débito',
        'debito': 'Débito',
        'crédito': 'Crédito',
        'credito': 'Crédito',
    }

    for pattern in PAYMENT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        raw = (match.group(1) or '').lower()
        return normalize.get(raw, raw.capitalize())

    return None


def extract_address(text: str) -> str | None:
    text_lower = text.lower().strip()

    # Remover ruído ANTES de tentar extrair endereço
    noise = [
        r'\b(pix|dinheiro|cart[aã]o|cartao|d[eé]bito|debito|cr[eé]dito|credito|dinheirinho)\b',
        r'\b(sem borda|borda de \w+)\b',
        r'\b(s[oó] isso|so isso|pode fechar|[eé] isso)\b',
        r'\b(vou pagar (no|na|com)|pagando (no|na|com)|forma de pagamento)\b',
        r'\b(no pix|na maquina|no dinheiro|no cartao)\b',
    ]
    text_clean = text_lower
    for pattern in noise:
        text_clean = re.sub(pattern, '', text_clean, flags=re.IGNORECASE)

    # Limpar espaços duplos e vírgulas soltas no final
    text_clean = re.sub(r',\s*$', '', text_clean.strip())
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    # Padrões de endereço — do mais específico ao mais genérico
    patterns = [
        # Padrão completo: rua x número y bairro z
        r"(rua|av|avenida|alameda|travessa|estrada)\s+\w+.{3,}",
        # Padrão médio: rua x número (pelo menos 8 chars totais)
        r"(rua|av|avenida|r\.)\s+[\w\s]+\s+\d+",
        # Padrão com bairro
        r"\w[\w\s]+\s+\d+[\w\s]*bairro\s+\w+",
        # Padrão mínimo: rua x 123
        r"(rua|av)\s+\w+\s+\d+",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            endereco = match.group(0).strip()
            # Limpar vírgulas e espaços soltos no final do endereço
            endereco = re.sub(r'[,\s]+$', '', endereco).strip()
            if len(endereco) >= 8:
                return endereco

    # Fallback: se tem "rua" no texto e comprimento razoável
    if re.search(r'\brua\b', text_clean, re.IGNORECASE) and len(text_clean) >= 8:
        clean = re.sub(r'[,\s]+$', '', text_clean).strip()
        return clean if len(clean) >= 8 else None

    return None


def is_affirmative(text: str) -> bool:
    stripped = text.strip()
    # Normalizar acentos para comparação
    text_norm = unicodedata.normalize('NFD', stripped.lower())
    text_norm = ''.join(c for c in text_norm if unicodedata.category(c) != 'Mn')
    
    # Testar patterns com texto normalizado
    if re.search(r'^(sim|s|isso|pode ser|certo|exato|confirmo|isso mesmo|pode|ta bom|ok|yes)[\s!.]*$', text_norm):
        return True
    
    if re.search(r'\b(confirmo|isso mesmo|pode ser|pode isso)\b', text_norm):
        return True
    
    if re.search(r'(ja falei|falei que sim|e sim|claro que sim|obvio)', text_norm):
        return True
    
    if re.search(r'(sim[\s,!]+sim|sim{2,})', text_norm):
        return True
    
    first = re.split(r'[\s,!.;:]+', text_norm, maxsplit=1)[0]
    return first in {'sim', 's', 'isso', 'ok', 'certo', 'confirmo'}

def extract_quantidade(text: str) -> int:
    """Extrai quantidade da mensagem. Padrão: 1."""
    word_to_num = {
        "uma": 1, "um": 1, "dois": 2, "duas": 2, "três": 3, "tres": 3,
        "quatro": 4, "cinco": 5, "meia dúzia": 6, "meia duzia": 6,
        "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10
    }
    text_lower = text.lower()
    
    for word, num in word_to_num.items():
        if word in text_lower:
            return num
    
    match = re.search(r'\b(\d{1,2})\b', text)
    if match:
        try:
            qty = int(match.group(1))
            if 1 <= qty <= 99:
                return qty
        except (ValueError, TypeError):
            pass
    
    return 1


def has_price_question(text: str) -> bool:
    """Detecta se o cliente perguntou sobre preço."""
    price_patterns = [
        r"quanto(s)? (fica|vai|custa|é|dar)",
        r"qual o (preço|valor|total)",
        r"quanto (fica|é isso|vai dar|sai)",
    ]
    text_lower = text.lower()
    for pattern in price_patterns:
        if re.search(pattern, text_lower):
            return True
    return False



def is_negative(text: str) -> bool:
    stripped = text.strip()
    if _extract_with_patterns(stripped, NEGATIVE):
        return True
    first = re.split(r'[\s,!.;:]+', stripped.lower(), maxsplit=1)[0]
    return first in {'nao', 'não', 'n', 'negativo', 'errado'}


def is_close_order(text: str) -> bool:
    return _extract_with_patterns(text, CLOSE_ORDER)


def _resolve_tamanho_label(session: dict[str, Any]) -> str:
    tamanho_display = _normalize_text(str(session.get('tamanho_display') or ''))
    tamanho_sigla = _normalize_text(str(session.get('tamanho') or ''))
    if tamanho_display:
        return tamanho_display
    return tamanho_sigla


def _get_borda_options(bordas: list[dict[str, Any]]) -> list[str]:
    opcoes = [str(item.get('nome') or '').strip() for item in bordas if str(item.get('nome') or '').strip()]
    if 'Sem borda' not in opcoes:
        opcoes.insert(0, 'Sem borda')
    return opcoes


def _get_borda_options_with_price(bordas: list[dict[str, Any]]) -> str:
    opcoes: list[str] = []
    for item in bordas:
        nome = str(item.get('nome') or '').strip()
        if not nome:
            continue
        try:
            preco_extra = float(item.get('preco_extra') or 0)
        except (TypeError, ValueError):
            preco_extra = 0.0
        if preco_extra > 0:
            preco_str = f'R${preco_extra:.0f}'.replace('.', ',')
            opcoes.append(f'{nome} (+{preco_str})')
        else:
            opcoes.append(nome)

    if not opcoes:
        return 'Sem borda'
    if not any('sem borda' == opt.lower() for opt in opcoes):
        opcoes.append('Sem borda')
    return ' | '.join(opcoes)


def _find_by_name(options: list[dict[str, Any]], field: str, value: str) -> dict[str, Any] | None:
    target = _normalize_text(value).lower()
    for item in options:
        current = _normalize_text(str(item.get(field) or '')).lower()
        if current == target:
            return item
    return None


def _get_sabor_price(sabor_obj: dict[str, Any] | None, tamanho_sigla: str, tamanho_obj: dict[str, Any] | None) -> float:
    if not sabor_obj:
        return 0.0

    precos = sabor_obj.get('precos')
    if isinstance(precos, dict):
        preco_tamanho = precos.get(tamanho_sigla)
        if preco_tamanho is not None:
            try:
                return float(preco_tamanho)
            except (TypeError, ValueError):
                pass

    try:
        preco_base = float(sabor_obj.get('preco_base') or 0)
    except (TypeError, ValueError):
        preco_base = 0.0

    try:
        modificador = float((tamanho_obj or {}).get('modificador_preco') or 0)
    except (TypeError, ValueError):
        modificador = 0.0

    return preco_base + modificador


def _montar_resumo(
    session: dict[str, Any],
    tamanhos: list[dict[str, Any]],
    bordas: list[dict[str, Any]],
    cardapio: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    sabor_nome = str(session.get('sabor') or session.get('sabor_sugerido') or 'Pizza').strip()
    tamanho_sigla = str(session.get('tamanho') or 'M').strip().upper()
    tamanho_label = _resolve_tamanho_label(session) or tamanho_sigla
    borda_nome = str(session.get('borda') or 'Sem borda').strip()
    endereco = str(session.get('endereco') or '').strip()
    pagamento = str(session.get('pagamento') or '').strip()

    sabor_obj = _find_by_name(cardapio, 'nome', sabor_nome)
    tamanho_obj = _find_by_name(tamanhos, 'sigla', tamanho_sigla)
    borda_obj = _find_by_name(bordas, 'nome', borda_nome) or {'preco_extra': 0}

    preco_sabor = _get_sabor_price(sabor_obj, tamanho_sigla, tamanho_obj)
    try:
        preco_borda = float(borda_obj.get('preco_extra') or 0)
    except (TypeError, ValueError):
        preco_borda = 0.0

    quantidade = int(session.get('quantidade', 1))
    preco_unitario = preco_sabor + preco_borda
    total = preco_unitario * quantidade
    session['total'] = total
    session['state'] = PizzaState.FINALIZADO.value

    borda_str = f' + {borda_nome}' if borda_nome and borda_nome.lower() != 'sem borda' else ''
    total_str = f'R${total:.2f}'.replace('.', ',')
    preco_unitario_str = f'R${preco_unitario:.2f}'.replace('.', ',')

    if quantidade > 1:
        item_str = f'🍕 {quantidade}x {sabor_nome} {tamanho_label}{borda_str} — {preco_unitario_str} cada'
    else:
        item_str = f'🍕 {sabor_nome} {tamanho_label}{borda_str}'

    resumo = (
        '✅ *Pedido confirmado!*\n\n'
        f'{item_str}\n'
        f'💰 *Total: {total_str}*\n\n'
        f'📍 {endereco}\n'
        f'💳 {pagamento}\n\n'
        'Tô separando já! Em breve saiu 🚀'
    )
    return resumo, session


def _avancar_apos_borda(
    session: dict[str, Any],
    tamanhos: list[dict[str, Any]],
    bordas: list[dict[str, Any]],
    cardapio: list[dict[str, Any]],
    text: str = "",
) -> tuple[str, dict[str, Any]]:
    if not session.get('tamanho'):
        session['tamanho'] = 'M'
        session['tamanho_display'] = 'M'

    # Tentar extrair endereço se ainda não tiver
    if not session.get('endereco') and text:
        endereco = extract_address(text)
        if endereco:
            session['endereco'] = endereco
    
    # Tentar extrair pagamento se ainda não tiver
    if not session.get('pagamento') and text:
        pagamento = extract_payment(text)
        if pagamento:
            session['pagamento'] = pagamento

    if not session.get('endereco'):
        session['state'] = PizzaState.AGUARDANDO_ENDERECO.value
        return 'Boa escolha! Endereço pra entrega? 📍', session

    if not session.get('pagamento'):
        session['state'] = PizzaState.AGUARDANDO_PAGAMENTO.value
        return 'Pix, dinheiro ou cartão? 💳', session

    return _montar_resumo(session, tamanhos, bordas, cardapio)


def process_pizza_message(
    text: str,
    session: dict[str, Any],
    cardapio: list[dict[str, Any]],
    tamanhos: list[dict[str, Any]],
    bordas: list[dict[str, Any]],
    tenant_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    state = session.get('state', PizzaState.AGUARDANDO_PEDIDO.value)

    logger.info(f"[PIZZA-FLOW] state atual: {session.get('state', 'NOVO')}")

    sabor_extraido = extract_sabor(text, cardapio) if not session.get('sabor') else None
    tamanho_extraido = extract_tamanho(text) if not session.get('tamanho') else None
    should_try_borda = (
        state == PizzaState.AGUARDANDO_BORDA.value
        or bool(re.search(r'\bborda\b|sem borda|rechead', text, re.IGNORECASE))
    )
    borda_extraida = (
        extract_borda(text, bordas)
        if (not session.get('borda') and should_try_borda)
        else None
    )
    pagamento_extraido = extract_payment(text) if not session.get('pagamento') else None
    endereco_extraido = extract_address(text) if not session.get('endereco') else None

    if sabor_extraido:
        session['sabor_sugerido'], session['sabor_score'] = sabor_extraido
    if tamanho_extraido:
        session['tamanho'], session['tamanho_display'] = tamanho_extraido
    if borda_extraida:
        session['borda'] = borda_extraida
    if pagamento_extraido:
        session['pagamento'] = pagamento_extraido
    if endereco_extraido:
        session['endereco'] = endereco_extraido

    if state == PizzaState.AGUARDANDO_PEDIDO.value:
        if not session.get('sabor_sugerido'):
            return 'Qual sabor de pizza você quer? 🍕', session

        # Extrair quantidade da mensagem inicial quando já houver sabor sugerido.
        qtd = extract_quantidade(text)
        if qtd > 1:
            session['quantidade'] = qtd

        sabor_nome = str(session.get('sabor_sugerido') or '').strip()
        tamanho_label = _resolve_tamanho_label(session)
        tamanho_str = f' {tamanho_label}' if tamanho_label else ''

        # SEMPRE passar por confirmação para capturar quantidade e dúvidas de preço.
        session['state'] = PizzaState.CONFIRMANDO_SABOR.value
        return f'Pizza de {sabor_nome}{tamanho_str}, certo? 🍕', session

    if state == PizzaState.CONFIRMANDO_SABOR.value:
        if is_negative(text):
            novo_sabor = extract_sabor(text, cardapio)
            if novo_sabor and novo_sabor[1] > 0.6:
                session['sabor_sugerido'] = novo_sabor[0]
                session['sabor_score'] = novo_sabor[1]
                # IMPORTANTE: não limpar o tamanho — preservar
                tamanho_label = _resolve_tamanho_label(session)
                tamanho_str = f' {tamanho_label}' if tamanho_label else ''
                # Manter em CONFIRMANDO_SABOR para reconfirmar
                return f'Pizza de {novo_sabor[0]}{tamanho_str}, certo? 🍕', session
            else:
                session.pop('sabor_sugerido', None)
                session.pop('sabor_score', None)
                session.pop('tamanho', None)
                session.pop('tamanho_display', None)
                session['state'] = PizzaState.AGUARDANDO_PEDIDO.value
                return 'Tudo bem! Qual sabor você quer então? 🍕', session

        should_confirm = is_affirmative(text) or (sabor_extraido and sabor_extraido[1] > 0.7)
        if should_confirm:
            session['sabor'] = str(session.get('sabor_sugerido') or '').strip()
            quantidade_extraida = extract_quantidade(text)
            quantidade = quantidade_extraida if quantidade_extraida > 1 else int(session.get('quantidade', 1) or 1)
            session['quantidade'] = quantidade
            price_asked = has_price_question(text)

            prefixo = ''
            if price_asked or quantidade > 1:
                sabor_obj = _find_by_name(cardapio, 'nome', session.get('sabor'))
                tamanho_sigla = str(session.get('tamanho') or 'M').strip().upper()
                tamanho_obj = _find_by_name(tamanhos, 'sigla', tamanho_sigla)
                preco_unitario = _get_sabor_price(sabor_obj, tamanho_sigla, tamanho_obj)
                preco_total = preco_unitario * quantidade
                preco_unitario_str = f'R${preco_unitario:.2f}'.replace('.', ',')
                preco_total_str = f'R${preco_total:.2f}'.replace('.', ',')
                sabor_nome = str(session.get('sabor') or '').strip()
                tamanho_label = _resolve_tamanho_label(session)
                if quantidade > 1:
                    prefixo = (
                        f'São {quantidade}x {sabor_nome} {tamanho_label} — '
                        f'{preco_unitario_str} cada, total {preco_total_str} (sem borda). '
                    )
                else:
                    prefixo = f'Fica {preco_unitario_str} (sem borda). '

            session['state'] = PizzaState.AGUARDANDO_BORDA.value

            if session.get('borda'):
                return _avancar_apos_borda(session, tamanhos, bordas, cardapio, text)

            opcoes_str = _get_borda_options_with_price(bordas)
            return f'{prefixo}Qual borda? {opcoes_str} 🧀', session

        novo_sabor = extract_sabor(text, cardapio)
        if novo_sabor:
            session['sabor_sugerido'], session['sabor_score'] = novo_sabor
            sabor_nome = str(novo_sabor[0]).strip()
            tamanho_label = _resolve_tamanho_label(session)
            borda_nome = str(session.get('borda') or '').strip()
            extras: list[str] = []
            if tamanho_label:
                extras.append(tamanho_label)
            if borda_nome:
                extras.append('sem borda' if borda_nome.lower() == 'sem borda' else borda_nome)
            extra_text = f" {' '.join(extras)}" if extras else ''
            return f'Pizza de {sabor_nome}{extra_text}, certo? 🍕', session

        return f"Confirma: pizza de {session.get('sabor_sugerido')}? 🍕", session

    if state == PizzaState.AGUARDANDO_BORDA.value:
        borda = extract_borda(text, bordas)
        if borda:
            session['borda'] = borda
            return _avancar_apos_borda(session, tamanhos, bordas, cardapio, text)

        opcoes_str = ' | '.join(_get_borda_options(bordas))
        return f'Qual borda você prefere? {opcoes_str}', session

    if state == PizzaState.AGUARDANDO_ENDERECO.value:
        if session.get('endereco'):
            # Tentar extrair pagamento da mesma mensagem se ainda não tiver
            if not session.get('pagamento'):
                _pag_inline = extract_payment(text)
                if _pag_inline:
                    session['pagamento'] = _pag_inline
            if not session.get('pagamento'):
                session['state'] = PizzaState.AGUARDANDO_PAGAMENTO.value
                return 'Pix, dinheiro ou cartão? 💳', session
            return _montar_resumo(session, tamanhos, bordas, cardapio)
        return 'Me manda o endereço completo pra entrega 📍', session

    if state == PizzaState.AGUARDANDO_PAGAMENTO.value:
        pagamento = extract_payment(text)
        if pagamento:
            session['pagamento'] = pagamento
            return _montar_resumo(session, tamanhos, bordas, cardapio)
        return 'Qual a forma de pagamento? Pix, dinheiro ou cartão 💳', session

    if state == PizzaState.FINALIZADO.value and is_close_order(text):
        return 'Pedido já fechado por aqui. Se quiser outro, me fala o sabor 🍕', session

    return 'Pode mandar seu pedido! 🍕', session
