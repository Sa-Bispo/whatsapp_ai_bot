from __future__ import annotations

import re
import logging
import unicodedata
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional
from order_receipt import gerar_comprovante

try:
    from rapidfuzz.fuzz import token_set_ratio  # pyright: ignore[reportMissingImports]
except Exception:
    def token_set_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return SequenceMatcher(None, a, b).ratio() * 100


logger = logging.getLogger(__name__)


class LanchoneteState(Enum):
    AGUARDANDO_PEDIDO    = "aguardando_pedido"
    CONFIRMANDO_ITEM     = "confirmando_item"
    AGUARDANDO_TAMANHO   = "aguardando_tamanho"
    AGUARDANDO_ADICIONAIS = "aguardando_adicionais"
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

_ORDER_INTENT_PATTERNS = re.compile(
    r'\b(quero|queria|me\s+ve|me\s+da|me\s+manda|vou\s+querer|traz|pede|pedir|coloca|colocar)\b',
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
    r"ta bom assim|mais não|mais nao|é tudo|isso é tudo)\b",
]

AFFIRMATIVE_PATTERNS = [
    r"^(sim|s|isso|pode ser|certo|exato|confirmo|isso mesmo|pode|"
    r"tá bom|ta bom|ok|yes|yeah|claro|obvio|óbvio)[\s!.]*$",
    r"\b(sim|confirmo|pode|isso|certo|exato)\b",
    r"(já falei|ja falei|falei que sim|é sim|e sim|claro que sim)",
    r"(sim[\s,!]+sim|sim+)",
    r"\bisso+\b",
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
    text_norm_soft = re.sub(r'[^a-z0-9]+', ' ', _norm(text_lower)).strip()
    text_clean = text_lower
    for sw in STOPWORDS_ITEM:
        text_clean = re.sub(sw, '', text_clean, flags=re.IGNORECASE).strip()
    text_clean_soft = re.sub(r'[^a-z0-9]+', ' ', _norm(text_clean)).strip()
    text_norm_soft = text_norm_soft.replace('bacao', 'bacon')
    text_clean_soft = text_clean_soft.replace('bacao', 'bacon')

    # Primeiro tenta um match direto por nome normalizado para mensagens longas
    # (ex.: "quero um x-bacon ... moro na rua ... pago no pix").
    for item in estoque:
        nome = str(item.get('nome') or '').strip()
        if not nome:
            continue
        nome_soft = re.sub(r'[^a-z0-9]+', ' ', _norm(nome)).strip()
        if not nome_soft:
            continue
        if nome_soft in text_norm_soft or nome_soft in text_clean_soft:
            return nome, 1.0

    # Match por token relevante ajuda em gírias/abreviações: "me vê um suco".
    if _ORDER_INTENT_PATTERNS.search(text_lower):
        for item in estoque:
            nome = str(item.get('nome') or '').strip()
            if not nome:
                continue
            nome_soft = re.sub(r'[^a-z0-9]+', ' ', _norm(nome)).strip()
            tokens = [
                t for t in nome_soft.split()
                if len(t) >= 4 and t not in {'porcao', 'frita', 'natural'}
            ]
            if any(re.search(rf'\b{re.escape(token)}\b', text_clean_soft) for token in tokens):
                return nome, 0.65

    best_match = None
    best_score = 0.0
    for item in estoque:
        nome = item['nome']
        score_orig = token_set_ratio(text_lower, nome.lower()) / 100
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


def extract_variacao(text: str, variacoes: list) -> Optional[dict]:
    """Extrai tamanho mencionado na mensagem."""
    text_norm = _norm(text)
    for v in variacoes:
        sigla = _norm(str(v.get('sigla', '') or ''))
        nome = _norm(str(v.get('nome', '') or ''))
        if sigla and re.search(rf'\b{sigla}\b', text_norm):
            return v
        if nome and re.search(rf'\b{nome}\b', text_norm):
            return v
    return None


def extract_adicionais(text: str, adicionais: list) -> list:
    """Extrai adicionais mencionados na mensagem."""
    text_norm = _norm(text)
    encontrados = []
    for a in adicionais:
        nome_norm = _norm(str(a.get('nome', '') or ''))
        score = token_set_ratio(text_norm, nome_norm) / 100
        if nome_norm and (score >= 0.7 or nome_norm in text_norm):
            encontrados.append(a)
    return encontrados


def has_no_adicional(text: str) -> bool:
    return bool(re.search(
        r'\b(sem adicional|sem nada|puro|s[oó]|nada|n[aã]o quero|nao quero)\b',
        _norm(text),
    ))


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
    ]
    for pattern in patterns:
        match = re.search(pattern, text_norm)
        if match:
            return int(match.group(1))

    # Evita interpretar número de endereço como quantidade.
    if re.search(r'\b(rua|av|avenida|alameda|travessa|estrada|bairro|numero|n\b|n\.|nº|n°)\b', text_norm):
        return 1

    match = re.search(r'\b(\d+)\b', text_norm)
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
        # Tolerantes: exigem indicador explícito de número de endereço
        r".{3,}\s+(n[°º.]?|n[uú]mero)\s*\d+",
        r"\w[\w\s]{3,},\s*(n[°º.]?\s*)?\d+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            end = re.sub(r'[,\s]+$', '', match.group(0)).strip()
            if len(end) >= 6:
                return end
    if re.search(r'\brua\b', text_clean, re.IGNORECASE) and len(text_clean) >= 6:
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


# ── helpers de formatação ────────────────────────────────────────

_TAMANHO_EMOJIS = {"P": "🔹", "M": "🔸", "G": "🔶", "GG": "🔥"}

PAGAMENTO_MSG = (
    "Como vai pagar? 💳\n\n"
    "💚 Pix\n"
    "💵 Dinheiro\n"
    "💳 Cartão"
)


def formatar_tamanhos(variacoes: list) -> str:
    linhas = []
    for v in variacoes:
        sigla = str(v.get("sigla") or "").strip()
        nome = str(v.get("nome") or sigla).strip()
        preco = f"R${float(v.get('preco', 0)):.2f}".replace(".", ",")
        emoji = _TAMANHO_EMOJIS.get(sigla.upper(), "•")
        label = f"{nome} ({sigla})" if nome and nome.upper() != sigla.upper() else sigla
        linhas.append(f"{emoji} {label} — {preco}")
    return "Qual tamanho? 🍟\n\n" + "\n".join(linhas)


def formatar_adicionais(adicionais: list) -> str:
    linhas = []
    for a in adicionais:
        preco = f"R${float(a.get('preco_extra', 0)):.2f}".replace(".", ",")
        linhas.append(f"➕ {a.get('nome', '')} — +{preco}")
    linhas.append("❌ Sem adicional")
    return "Quer algum adicional? 🍟\n\n" + "\n".join(linhas)


def formatar_confirmacao_item(
    nome: str,
    variacao: Optional[dict],
    adicionais: list,
    preco: float,
) -> str:
    sigla = str((variacao or {}).get("sigla") or "").strip()
    nome_completo = f"{nome} {sigla}".strip() if sigla else nome
    linhas = ["✅ Anotado!\n", f"🍔 *{nome_completo}*"]
    for a in adicionais:
        linhas.append(f"   ➕ {a.get('nome', '')}")
    preco_fmt = f"R${preco:.2f}".replace(".", ",")
    linhas.append(f"\n💰 *{preco_fmt}*")
    return "\n".join(linhas)


def mais_itens_com_carrinho(session: dict) -> str:
    carrinho = session.get("carrinho", [])

    if not carrinho:
        return "Quer mais alguma coisa ou pode fechar? 😊"

    linhas = ["🛒 *No seu pedido até agora:*\n"]
    total = 0.0
    for item in carrinho:
        qty = item["quantidade"]
        preco_unit = item["preco"]
        subtotal = preco_unit * qty
        total += subtotal
        preco_fmt = f"R${subtotal:.2f}".replace(".", ",")
        qty_str = f"{qty}x " if qty > 1 else ""
        linhas.append(f"• {qty_str}{item['nome']} — {preco_fmt}")

    total_fmt = f"R${total:.2f}".replace(".", ",")
    linhas.append(f"\n💰 *Total parcial: {total_fmt}*")
    linhas.append("\nQuer mais alguma coisa ou pode fechar? 😊")
    return "\n".join(linhas)


# ── engine principal ─────────────────────────────────────────────

def process_lanchonete_message(
    text: str,
    session: dict,
    estoque: list[dict],
    tenant_config: dict,
) -> tuple[str | list[str] | None, dict]:
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
            item_obj = next((i for i in estoque if i.get('nome') == nome), None)
            session['item_sugerido'] = nome
            session['produto_consultado'] = nome
            session['quantidade_sugerida'] = extract_quantidade(text)
            session['fechar_apos_confirmacao'] = is_close_order(text)

            if item_obj and item_obj.get('tem_variacoes') and item_obj.get('variacoes'):
                variacao = extract_variacao(text, item_obj.get('variacoes', []))
                if variacao:
                    session['variacao_sugerida'] = variacao
                adicionais = extract_adicionais(text, item_obj.get('adicionais', []))
                if adicionais:
                    session['adicionais_sugeridos'] = adicionais

            session['state'] = LanchoneteState.CONFIRMANDO_ITEM.value
            qty = session['quantidade_sugerida']
            qty_str = f"{qty}x " if qty > 1 else ""
            return f"{qty_str}{nome}, certo? 🍔", session

        produto_contexto = str(session.get('produto_consultado') or '').strip()
        if produto_contexto and (
            _ORDER_INTENT_PATTERNS.search(text)
            or is_affirmative(text)
            or re.search(r'\b(um|uma|entao|então)\b', _norm(text))
        ):
            session['item_sugerido'] = produto_contexto
            session['quantidade_sugerida'] = extract_quantidade(text)
            session['state'] = LanchoneteState.CONFIRMANDO_ITEM.value
            qty = session['quantidade_sugerida']
            qty_str = f"{qty}x " if qty > 1 else ""
            return f"{qty_str}{produto_contexto}, certo? 🍔", session

        if session.get('endereco') or session.get('pagamento'):
            return 'Perfeito, já anotei os dados da entrega. Qual item você quer pedir? 🍔', session
        return 'Me diz o que você quer pedir que eu já anoto aqui 🍔', session

    # ── CONFIRMANDO_ITEM ─────────────────────────────────────────
    if state == LanchoneteState.CONFIRMANDO_ITEM.value:
        confirmou_item = is_affirmative(text) or (
            (session.get('endereco') or session.get('pagamento')) and not is_negative(text)
        )
        if confirmou_item:
            nome = session.get('item_sugerido', '')
            item_obj = next((i for i in estoque if i.get('nome') == nome), None)
            qty_new = extract_quantidade(text)
            qty = qty_new if (qty_new > 1 or 'quero' in text.lower()) else session.get('quantidade_sugerida', 1)
            session['quantidade_atual'] = qty

            if item_obj and item_obj.get('tem_variacoes') and item_obj.get('variacoes'):
                tamanho = extract_variacao(text, item_obj.get('variacoes', [])) or session.pop('variacao_sugerida', None)
                if tamanho:
                    session['variacao_atual'] = tamanho
                    adicionais = extract_adicionais(text, item_obj.get('adicionais', []))
                    if not adicionais:
                        adicionais = session.pop('adicionais_sugeridos', [])
                    session['adicionais_atuais'] = adicionais
                    if item_obj.get('adicionais') and not adicionais and not has_no_adicional(text):
                        session['state'] = LanchoneteState.AGUARDANDO_ADICIONAIS.value
                        return formatar_adicionais(item_obj.get('adicionais', [])), session
                    return _confirmar_item_com_variacoes(session, item_obj)

                session['state'] = LanchoneteState.AGUARDANDO_TAMANHO.value
                return formatar_tamanhos(item_obj.get('variacoes', [])), session

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

            if (
                is_close_order(text)
                or session.pop('fechar_apos_confirmacao', False)
                or (session.get('endereco') and session.get('pagamento'))
            ):
                return _advance_from_mais_itens(session, estoque, prefixo, tenant_config)

            total_item = preco * qty
            preco_fmt = f"R${total_item:.2f}".replace('.', ',')
            msg1_parts = []
            if prefixo:
                msg1_parts.append(prefixo.strip())
            msg1_parts.append('✅ Anotado!')
            msg1_parts.append(f"🍔 *{nome}*")
            msg1_parts.append(f"💰 *{preco_fmt}*")
            msg1 = '\n'.join(msg1_parts)

            session['state'] = LanchoneteState.MAIS_ITENS.value
            msg2 = mais_itens_com_carrinho(session)
            return [msg1, msg2], session
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

    # ── AGUARDANDO_TAMANHO ───────────────────────────────────────
    if state == LanchoneteState.AGUARDANDO_TAMANHO.value:
        item_obj = next((i for i in estoque if i.get('nome') == session.get('item_sugerido')), None)
        variacoes = item_obj.get('variacoes', []) if item_obj else []
        adicionais_disponiveis = item_obj.get('adicionais', []) if item_obj else []

        tamanho = extract_variacao(text, variacoes)
        if tamanho:
            session['variacao_atual'] = tamanho
            adicionais = extract_adicionais(text, adicionais_disponiveis)
            session['adicionais_atuais'] = adicionais

            if adicionais_disponiveis and not adicionais and not has_no_adicional(text):
                session['state'] = LanchoneteState.AGUARDANDO_ADICIONAIS.value
                return formatar_adicionais(adicionais_disponiveis), session

            return _confirmar_item_com_variacoes(session, item_obj)

        return formatar_tamanhos(variacoes), session

    # ── AGUARDANDO_ADICIONAIS ────────────────────────────────────
    if state == LanchoneteState.AGUARDANDO_ADICIONAIS.value:
        item_obj = next((i for i in estoque if i.get('nome') == session.get('item_sugerido')), None)
        adicionais_disponiveis = item_obj.get('adicionais', []) if item_obj else []

        if has_no_adicional(text):
            session['adicionais_atuais'] = []
        else:
            adicionais = extract_adicionais(text, adicionais_disponiveis)
            session['adicionais_atuais'] = adicionais

        return _confirmar_item_com_variacoes(session, item_obj)

    # ── MAIS_ITENS ───────────────────────────────────────────────
    if state == LanchoneteState.MAIS_ITENS.value:
        if is_close_order(text):
            return _advance_from_mais_itens(session, estoque, '', tenant_config)

        # Cliente pode enviar dados de checkout sem repetir "pode fechar".
        if session.get('endereco') or session.get('pagamento'):
            return _advance_from_mais_itens(session, estoque, '', tenant_config)

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
            return _advance_from_mais_itens(session, estoque, '', tenant_config)

        # Mantém fluxo determinístico para evitar fallback de IA em conversa de checkout.
        return mais_itens_com_carrinho(session), session

    # ── AGUARDANDO_ENDERECO ──────────────────────────────────────
    if state == LanchoneteState.AGUARDANDO_ENDERECO.value:
        if session.get('endereco'):
            if session.get('pagamento'):
                return _montar_resumo(session, estoque, '', tenant_config)
            session['state'] = LanchoneteState.AGUARDANDO_PAGAMENTO.value
            return PAGAMENTO_MSG, session
        return "Me manda o endereço completo pra entrega 📍", session

    # ── AGUARDANDO_PAGAMENTO ───────────────────────────────────────
    if state == LanchoneteState.AGUARDANDO_PAGAMENTO.value:
        if session.get('pagamento'):
            return _montar_resumo(session, estoque, '', tenant_config)
        return PAGAMENTO_MSG, session

    return 'Me fala seu pedido que eu te ajudo a fechar rapidinho 🍔', session


def _advance_from_mais_itens(
    session: dict,
    estoque: list[dict],
    prefixo: str = '',
    tenant_config: dict | None = None,
) -> tuple[str, dict]:
    if session.get('endereco') and session.get('pagamento'):
        return _montar_resumo(session, estoque, prefixo, tenant_config)
    elif session.get('endereco'):
        session['state'] = LanchoneteState.AGUARDANDO_PAGAMENTO.value
        return prefixo + PAGAMENTO_MSG, session
    else:
        session['state'] = LanchoneteState.AGUARDANDO_ENDERECO.value
        return f"{prefixo}Show! Qual o endereço pra entrega? 📍", session


def _montar_resumo(
    session: dict,
    estoque: list[dict],
    prefixo: str = '',
    tenant_config: dict | None = None,
) -> tuple[str, dict]:
    carrinho = session.get('carrinho', [])
    endereco = session.get('endereco', '')
    pagamento = session.get('pagamento', '')
    total = 0.0
    itens: list[dict] = []
    for item in carrinho:
        qty   = item['quantidade']
        nome  = item['nome']
        preco = item['preco']
        itens.append({'nome': nome, 'preco': preco, 'quantidade': qty})
        total += preco * qty
    nome_negocio = str((tenant_config or {}).get('nome_negocio') or 'Lanchonete').strip() or 'Lanchonete'

    session['state'] = LanchoneteState.FINALIZADO.value
    session['total'] = total

    comprovante = gerar_comprovante(
        itens=itens,
        total=total,
        endereco=str(endereco or ''),
        pagamento=str(pagamento or ''),
        nome_negocio=nome_negocio,
        sub_nicho='lanchonete',
    )
    return prefixo + comprovante, session


def _confirmar_item_com_variacoes(session: dict, item_obj: dict) -> tuple[list[str], dict]:
    variacao = session.get('variacao_atual', {})
    adicionais = session.get('adicionais_atuais', [])
    nome_item = session.get('item_sugerido', '')

    preco_base = float(variacao.get('preco', item_obj.get('preco', 0)) or 0)
    preco_adicionais = sum(float(a.get('preco_extra', 0) or 0) for a in adicionais)
    preco_total = preco_base + preco_adicionais

    partes = [f"{nome_item} {variacao.get('sigla', '')}".strip()]
    for a in adicionais:
        partes.append(f"+ {a.get('nome', '')}")
    nome_completo = ' '.join(partes).strip()

    if 'carrinho' not in session:
        session['carrinho'] = []

    qty = session.get('quantidade_atual', session.get('quantidade_sugerida', 1))
    session['carrinho'].append({
        'nome': nome_completo,
        'quantidade': qty,
        'preco': preco_total,
    })

    session.pop('item_sugerido', None)
    session.pop('quantidade_sugerida', None)
    session.pop('variacao_atual', None)
    session.pop('adicionais_atuais', None)
    session.pop('variacao_sugerida', None)
    session.pop('adicionais_sugeridos', None)
    session.pop('quantidade_atual', None)

    session['state'] = LanchoneteState.MAIS_ITENS.value
    confirmacao = formatar_confirmacao_item(nome_item, variacao, adicionais, preco_total)
    carrinho_str = mais_itens_com_carrinho(session)
    return [confirmacao, carrinho_str], session


def save_lanchonete_order_payload(session: dict) -> dict:
    return {
        'carrinho': session.get('carrinho', []),
        'total': session.get('total', 0.0),
        'endereco': session.get('endereco', ''),
        'pagamento': session.get('pagamento', ''),
    }
