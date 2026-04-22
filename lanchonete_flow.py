from __future__ import annotations

import re
import logging
import unicodedata
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional

try:
    from rapidfuzz.fuzz import token_set_ratio  # pyright: ignore[reportMissingImports]
except Exception:
    def token_set_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return SequenceMatcher(None, a, b).ratio() * 100


logger = logging.getLogger(__name__)


class LanchoneteState(Enum):
    AGUARDANDO_PEDIDO    = "aguardando_pedido"
    CONFIRMANDO_ITEM     = "confirmando_item"
    MAIS_ITENS           = "mais_itens"
    AGUARDANDO_ENDERECO  = "aguardando_endereco"
    AGUARDANDO_PAGAMENTO = "aguardando_pagamento"
    FINALIZADO           = "finalizado"


# ── helpers de normalização ──────────────────────────────────────

STOPWORDS_ITEM = [
    r"^(opa|oi|olá|ola|bom dia|boa tarde|boa noite)[,\s]+",
    r"\b(tem|temos|vocês têm|voces tem|vocês tem)\s+",
    r"\b(me vê|me da|me dá|quero|queria|pode ser|vou querer|traz)\s+",
    r"\b(aí|ai|lá|la|por favor|pf|pfv)\b",
]

_INFORMATIONAL_PATTERNS = re.compile(
    r'\b(diferença|diferenca|comparação|comparacao|melhor|pior|recomend|indica|'
    r'me explica|explica|como é|como e|como fica|vale a pena|qual o melhor|'
    r'me fala sobre|me conta|o que é|o que e|ingrediente|calorias|valor nutricional)\b',
    re.IGNORECASE,
)

PAYMENT_MAP = {
    "pix": "Pix", "dinheiro": "Dinheiro", "dinheirinho": "Dinheiro",
    "cartão": "Cartão", "cartao": "Cartão",
    "débito": "Débito", "debito": "Débito",
    "crédito": "Crédito", "credito": "Crédito",
    "maquina": "Cartão", "máquina": "Cartão",
}

CLOSE_ORDER_PATTERNS = [
    r"\b(só isso|so isso|é isso|e isso|pode fechar|fecha|só isso mesmo|"
    r"so isso mesmo|só isso só|por enquanto é isso|tá bom assim|"
    r"ta bom assim|mais não|mais nao|é tudo|e tudo)\b",
]

AFFIRMATIVE_PATTERNS = [
    r"^(sim|s|isso|pode ser|certo|exato|confirmo|isso mesmo|pode|"
    r"tá bom|ta bom|ok|yes|yeah|claro|obvio|óbvio)[\s!.]*$",
    r"\b(sim|confirmo|pode|isso|certo|exato)\b",
    r"(já falei|ja falei|falei que sim|é sim|e sim|claro que sim)",
    r"(sim[\s,!]+sim|sim+)",
]

NEGATIVE_PATTERNS = [
    r"^(não|nao|n|negativo|errado|outro|diferente)[\s!.]*$",
]

ADDRESS_NOISE = [
    r"\b(pix|dinheiro|cartão|cartao|débito|debito|crédito|credito|dinheirinho|maquina|máquina)\b",
    r"\b(só isso|so isso|pode fechar|é isso)\b",
    r"\b(vou pagar (no|na|com)|pagando (no|na|com))\b",
    r"\b(no pix|na maquina|no dinheiro|no cartao|no cartão)\b",
]


def _norm(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )


def extract_produto(text: str, estoque: list[dict]) -> Optional[tuple[str, float]]:
    if _INFORMATIONAL_PATTERNS.search(text):
        return None

    text_lower = text.lower().strip()
    text_clean = text_lower
    for sw in STOPWORDS_ITEM:
        text_clean = re.sub(sw, '', text_clean, flags=re.IGNORECASE).strip()

    best_match = None
    best_score = 0.0
    for item in estoque:
        if not item.get('disponivel', True):
            continue
        nome = item['nome']
        score_orig  = token_set_ratio(text_lower, nome.lower()) / 100
        score_clean = token_set_ratio(text_clean, nome.lower()) / 100
        score = max(score_orig, score_clean)
        if score > best_score:
            best_score = score
            best_match = nome

    if best_score >= 0.45:
        return best_match, best_score
    return None


def _get_preco(nome: str, estoque: list[dict]) -> float:
    item = next((i for i in estoque if i['nome'] == nome), None)
    if item:
        return float(item.get('preco', item.get('preco_base', 0)) or 0)
    return 0.0


def _add_to_cart(session: dict, nome: str, quantidade: int, preco: float) -> None:
    carrinho = session.get('carrinho', [])
    existing = next((i for i in carrinho if i['nome'] == nome), None)
    if existing:
        existing['quantidade'] += quantidade
    else:
        carrinho.append({'nome': nome, 'quantidade': quantidade, 'preco': preco})
    session['carrinho'] = carrinho


def extract_quantidade(text: str) -> int:
    word_to_num = {
        'um': 1, 'uma': 1, 'dois': 2, 'duas': 2,
        'três': 3, 'tres': 3, 'quatro': 4, 'cinco': 5,
        'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9, 'dez': 10,
    }
    text_norm = _norm(text)
    for word, num in word_to_num.items():
        if re.search(rf'\b{word}\b', text_norm):
            return num
    patterns = [
        r'(?:sim|pode|certo|isso|quero)[,\s]+(?:quero\s+)?(\d+)',
        r'(?:quero|me\s+manda|me\s+da|me\s+ve)\s+(\d+)',
        r'\b(\d+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_norm)
        if match:
            return int(match.group(1))
    return 1


def extract_pagamento(text: str) -> Optional[str]:
    text_norm = _norm(text)
    for key, val in PAYMENT_MAP.items():
        if re.search(rf'\b{_norm(key)}\b', text_norm):
            return val
    return None


def extract_endereco(text: str) -> Optional[str]:
    text_lower = text.lower().strip()
    text_clean = text_lower
    for pattern in ADDRESS_NOISE:
        text_clean = re.sub(pattern, '', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r',\s*$', '', text_clean.strip())
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    patterns = [
        r"(rua|av|avenida|alameda|travessa|estrada)\s+\w+.{3,}",
        r"(rua|av|avenida|r\.)\s+[\w\s]+\s+\d+",
        r"\w[\w\s]+\s+\d+[\w\s]*bairro\s+\w+",
        r"(rua|av)\s+\w+\s+\d+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            end = re.sub(r'[,\s]+$', '', match.group(0)).strip()
            if len(end) >= 8:
                return end
    if re.search(r'\brua\b', text_clean, re.IGNORECASE) and len(text_clean) >= 8:
        return re.sub(r'[,\s]+$', '', text_clean).strip()
    return None


def is_affirmative(text: str) -> bool:
    text_norm = _norm(text.strip())
    for p in AFFIRMATIVE_PATTERNS:
        if re.search(p, text_norm, re.IGNORECASE):
            return True
    return False


def is_negative(text: str) -> bool:
    text_norm = _norm(text.strip())
    for p in NEGATIVE_PATTERNS:
        if re.search(p, text_norm, re.IGNORECASE):
            return True
    return False


def is_close_order(text: str) -> bool:
    text_norm = _norm(text)
    for p in CLOSE_ORDER_PATTERNS:
        if re.search(p, text_norm, re.IGNORECASE):
            return True
    return False


def has_price_question(text: str) -> bool:
    return bool(re.search(
        r'(quanto(s)?\s+(fica|vai|custa|é)|qual\s+o\s+(preço|valor|total)|quanto\s+(fica|é\s+isso))',
        text.lower(),
    ))


# ── engine principal ─────────────────────────────────────────────

def process_lanchonete_message(
    text: str,
    session: dict,
    estoque: list[dict],
    tenant_config: dict,
) -> tuple[str | None, dict]:
    """
    Processa mensagem no fluxo de lanchonete com carrinho acumulado.
    Retorna (resposta, session_atualizada).
    Retorna (None, session) quando não reconhecido — cair no fallback de IA.
    """
    state = session.get('state', LanchoneteState.AGUARDANDO_PEDIDO.value)

    # Extração antecipada de pagamento e endereço em qualquer mensagem
    if not session.get('pagamento'):
        pag = extract_pagamento(text)
        if pag:
            session['pagamento'] = pag

    if not session.get('endereco'):
        end = extract_endereco(text)
        if end:
            session['endereco'] = end

    # ── AGUARDANDO_PEDIDO ────────────────────────────────────────
    if state == LanchoneteState.AGUARDANDO_PEDIDO.value:
        resultado = extract_produto(text, estoque)
        if resultado:
            nome, _score = resultado
            session['item_sugerido'] = nome
            session['quantidade_sugerida'] = extract_quantidade(text)
            session['state'] = LanchoneteState.CONFIRMANDO_ITEM.value
            qty = session['quantidade_sugerida']
            qty_str = f"{qty}x " if qty > 1 else ""
            return f"{qty_str}{nome}, certo? 🍔", session
        return None, session  # fallback IA

    # ── CONFIRMANDO_ITEM ─────────────────────────────────────────
    if state == LanchoneteState.CONFIRMANDO_ITEM.value:
        if is_affirmative(text):
            nome = session.get('item_sugerido', '')
            qty_new = extract_quantidade(text)
            qty = qty_new if (qty_new > 1 or 'quero' in text.lower()) else session.get('quantidade_sugerida', 1)
            preco = _get_preco(nome, estoque)

            prefixo = ''
            if has_price_question(text):
                total_item = preco * qty
                preco_str = f"R${preco:.2f}".replace('.', ',')
                total_str = f"R${total_item:.2f}".replace('.', ',')
                prefixo = f"Fica {total_str} ({qty}x {preco_str}). " if qty > 1 else f"Fica {preco_str}. "

            _add_to_cart(session, nome, qty, preco)
            session.pop('item_sugerido', None)
            session.pop('quantidade_sugerida', None)

            if is_close_order(text):
                return _advance_from_mais_itens(session, estoque, prefixo)

            session['state'] = LanchoneteState.MAIS_ITENS.value
            return f"{prefixo}Mais alguma coisa? Ou pode fechar? 😊", session

        elif is_negative(text):
            novo = extract_produto(text, estoque)
            if novo:
                session['item_sugerido'] = novo[0]
                session['quantidade_sugerida'] = extract_quantidade(text)
                qty = session['quantidade_sugerida']
                qty_str = f"{qty}x " if qty > 1 else ""
                return f"{qty_str}{novo[0]}, certo? 🍔", session
            session.pop('item_sugerido', None)
            session['state'] = LanchoneteState.AGUARDANDO_PEDIDO.value
            return "Tudo bem! O que você quer pedir? 🍔", session

        else:
            novo = extract_produto(text, estoque)
            if novo:
                session['item_sugerido'] = novo[0]
                session['quantidade_sugerida'] = extract_quantidade(text)
                qty = session['quantidade_sugerida']
                qty_str = f"{qty}x " if qty > 1 else ""
                return f"{qty_str}{novo[0]}, certo? 🍔", session
            qty = extract_quantidade(text)
            if qty > 1:
                session['quantidade_sugerida'] = qty
                return f"{qty}x {session.get('item_sugerido', '')}, certo? 🍔", session
            return f"Confirma: {session.get('item_sugerido')}? 🍔", session

    # ── MAIS_ITENS ───────────────────────────────────────────────
    if state == LanchoneteState.MAIS_ITENS.value:
        if is_close_order(text):
            return _advance_from_mais_itens(session, estoque)

        novo = extract_produto(text, estoque)
        if novo:
            session['item_sugerido'] = novo[0]
            session['quantidade_sugerida'] = extract_quantidade(text)
            session['state'] = LanchoneteState.CONFIRMANDO_ITEM.value
            qty = session['quantidade_sugerida']
            qty_str = f"{qty}x " if qty > 1 else ""
            return f"{qty_str}{novo[0]}, certo? 🍔", session

        # "não" / "nao" sem produto = encerrar pedido
        if is_negative(text):
            return _advance_from_mais_itens(session, estoque)

        return "O que mais? Ou me manda 'só isso' pra fechar 😊", session

    # ── AGUARDANDO_ENDERECO ──────────────────────────────────────
    if state == LanchoneteState.AGUARDANDO_ENDERECO.value:
        if session.get('endereco'):
            if session.get('pagamento'):
                return _montar_resumo(session, estoque)
            session['state'] = LanchoneteState.AGUARDANDO_PAGAMENTO.value
            return "Pix, dinheiro ou cartão? 💳", session
        return "Me manda o endereço completo pra entrega 📍", session

    # ── AGUARDANDO_PAGAMENTO ─────────────────────────────────────
    if state == LanchoneteState.AGUARDANDO_PAGAMENTO.value:
        if session.get('pagamento'):
            return _montar_resumo(session, estoque)
        return "Qual a forma de pagamento? Pix, dinheiro ou cartão 💳", session

    return None, session


def _advance_from_mais_itens(
    session: dict,
    estoque: list[dict],
    prefixo: str = '',
) -> tuple[str, dict]:
    if session.get('endereco') and session.get('pagamento'):
        return _montar_resumo(session, estoque, prefixo)
    elif session.get('endereco'):
        session['state'] = LanchoneteState.AGUARDANDO_PAGAMENTO.value
        return f"{prefixo}Pix, dinheiro ou cartão? 💳", session
    else:
        session['state'] = LanchoneteState.AGUARDANDO_ENDERECO.value
        return f"{prefixo}Show! Qual o endereço pra entrega? 📍", session


def _montar_resumo(session: dict, estoque: list[dict], prefixo: str = '') -> tuple[str, dict]:
    carrinho = session.get('carrinho', [])
    endereco = session.get('endereco', '')
    pagamento = session.get('pagamento', '')

    linhas: list[str] = []
    total = 0.0
    for i, item in enumerate(carrinho):
        qty   = item['quantidade']
        nome  = item['nome']
        preco = item['preco']
        preco_str = f"R${preco:.2f}".replace('.', ',')
        if qty > 1:
            linha = f"{qty}x {nome} — {preco_str} cada"
        else:
            linha = f"1x {nome} — {preco_str}"
        linhas.append(f"🛒 {linha}" if i == 0 else f"   {linha}")
        total += preco * qty

    total_str = f"R${total:.2f}".replace('.', ',')
    itens_str = '\n'.join(linhas)

    session['state'] = LanchoneteState.FINALIZADO.value
    session['total'] = total

    resumo = (
        f"✅ *Pedido anotado!*\n\n"
        f"{itens_str}\n"
        f"💰 *Total: {total_str}*\n\n"
        f"📍 {endereco}\n"
        f"💳 {pagamento}\n\n"
        "Tô separando já! 🚀"
    )
    return prefixo + resumo, session


def save_lanchonete_order_payload(session: dict) -> dict:
    return {
        'carrinho': session.get('carrinho', []),
        'total': session.get('total', 0.0),
        'endereco': session.get('endereco', ''),
        'pagamento': session.get('pagamento', ''),
    }
