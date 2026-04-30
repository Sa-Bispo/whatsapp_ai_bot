from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from enum import Enum
from typing import Any
from order_receipt import gerar_comprovante

try:
    from rapidfuzz.fuzz import token_set_ratio  # pyright: ignore[reportMissingImports]
except Exception:
    def token_set_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100


logger = logging.getLogger(__name__)


class PizzaState(Enum):
    AGUARDANDO_PEDIDO = 'aguardando_pedido'
    CONFIRMANDO_SABOR = 'confirmando_sabor'
    AGUARDANDO_TAMANHO = 'aguardando_tamanho'
    AGUARDANDO_BORDA = 'aguardando_borda'
    MAIS_PIZZAS = 'mais_pizzas'
    OFERECENDO_BEBIDAS = 'oferecendo_bebidas'
    CONFIRMANDO_BEBIDA = 'confirmando_bebida'
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

NEGATIVE = [
    r'^(n[aã]o|nao|n|negativo|errado|outro|diferente)[\s!.]*$',
]

CLOSE_ORDER_PATTERNS = [
    r"\b(s[oó] isso|so isso|[eé] isso|e isso|pode fechar|fecha|s[oó] isso mesmo|"
    r"so isso mesmo|s[oó] isso s[oó]|por enquanto [eé] isso|t[aá] bom assim|"
    r"ta bom assim|mais n[aã]o|mais nao|[eé] tudo|e tudo|pode|"
    r"quero mais nada|nao quero mais|n[aã]o quero mais|"
    r"mais nada|nada mais|t[aá] [oó]timo|ta otimo|"
    r"s[oó] as pizzas|so as pizzas|apenas as pizzas)\b"
]

RECUSA_BEBIDA_PATTERNS = [
    r"\b(n[aã]o|nao|n|sem bebida|s[oó] pizzas|so pizzas|apenas as pizzas|"
    r"pode fechar|s[oó] isso|so isso|n[aã]o quero|nao quero|"
    r"t[aá] bom|ta bom|t[aá] [oó]timo|ta otimo|s[oó] as pizzas|so as pizzas|"
    r"mn|mano|n[aã]o precisa|nao precisa)\b"
]


def _normalize_text(value: str) -> str:
    return (value or '').strip()


def _normalize_for_match(value: str) -> str:
    raw = unicodedata.normalize('NFKD', value or '')
    without_accents = ''.join(ch for ch in raw if not unicodedata.combining(ch))
    lowered = without_accents.lower()
    return re.sub(r'[^a-z0-9\s]+', ' ', lowered).strip()


def _norm(value: str) -> str:
    return _normalize_for_match(value)


def _extract_with_patterns(text: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _ensure_session_defaults(session: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(session, dict):
        session = {}

    session.setdefault('state', PizzaState.AGUARDANDO_PEDIDO.value)
    session.setdefault('carrinho', [])
    session.setdefault('fila_sabores', [])
    session.setdefault('pizza_atual', {})

    if not isinstance(session.get('carrinho'), list):
        session['carrinho'] = []
    if not isinstance(session.get('fila_sabores'), list):
        session['fila_sabores'] = []
    if not isinstance(session.get('pizza_atual'), dict):
        session['pizza_atual'] = {}

    return session


def _new_pizza(sabor: str) -> dict[str, Any]:
    return {
        'sabor': sabor,
        'tamanho': None,
        'borda': None,
    }


def extract_tamanho(text: str) -> tuple[str, str] | None:
    for tamanho in ('GG', 'G', 'M', 'P'):
        for pattern in TAMANHO_PATTERNS[tamanho]:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            raw = (match.group(0) or '').strip()
            if raw and raw.upper() in {'P', 'M', 'G', 'GG'}:
                return tamanho, raw.upper()
            return tamanho, tamanho
    return None


def _limpar_stopwords(text: str) -> str:
    text_clean = _normalize_for_match(text)
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
        r'^quero\s+duas[,\s]*',
        r'^quero\s+duas\s+pizzas[,\s]*',
        r'^duas\s+pizzas[,\s]*',
    ]
    for sw in stopwords:
        text_clean = re.sub(sw, '', text_clean, flags=re.IGNORECASE).strip()
    return text_clean


def extract_sabor(text: str, cardapio: list[dict[str, Any]]) -> tuple[str, float] | None:
    logger.info(f"[PIZZA] extract_sabor input: '{text}'")

    text_lower = _normalize_for_match(text).strip()
    if not text_lower:
        return None

    text_clean = _limpar_stopwords(text_lower)
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

        if score > best_score:
            best_score = score
            best_match = nome

    if best_match and best_score >= 0.60:
        return best_match, best_score
    return None


def extract_todos_sabores(text: str, cardapio: list[dict[str, Any]]) -> list[str]:
    encontrados: list[str] = []
    text_clean = _limpar_stopwords(text)

    partes = re.split(r'\b(e|,|uma de|outra de|mais uma)\b', text_clean, flags=re.IGNORECASE)
    for parte in partes:
        pedaco = (parte or '').strip()
        if not pedaco:
            continue

        resultado = extract_sabor(pedaco, cardapio)
        if resultado and resultado[1] >= 0.45 and resultado[0] not in encontrados:
            encontrados.append(resultado[0])

    if encontrados:
        return encontrados

    resultado_unico = extract_sabor(text, cardapio)
    if resultado_unico:
        return [resultado_unico[0]]

    return []


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


def extract_quantidade(text: str) -> int:
    word_to_num = {
        'uma': 1,
        'um': 1,
        'duas': 2,
        'dois': 2,
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
    text_lower = _normalize_for_match(text)
    for word, num in word_to_num.items():
        if re.search(rf'\b{word}\b', text_lower):
            return num

    match = re.search(r'\b(\d{1,2})\b', text_lower)
    if match:
        try:
            value = int(match.group(1))
            if 1 <= value <= 99:
                return value
        except (TypeError, ValueError):
            pass
    return 1


def extract_bebida(text: str, bebidas: list[dict[str, Any]]) -> tuple[str, float] | None:
    text_lower = (text or '').lower().strip()
    text_clean = re.sub(r'\b(quero|uma|um|pode ser|me da|me dá|traz)\b', '', text_lower).strip()

    best_match = None
    best_score = 0.0
    best_preco = 0.0

    for bebida in bebidas:
        if not bebida.get('disponivel', True):
            continue

        nome = str(bebida.get('nome') or '').strip()
        if not nome:
            continue

        score_orig = token_set_ratio(text_lower, nome.lower()) / 100
        score_clean = token_set_ratio(text_clean, nome.lower()) / 100
        score = max(score_orig, score_clean)

        if score > best_score:
            best_score = score
            best_match = nome
            try:
                best_preco = float(bebida.get('preco') or 0)
            except (TypeError, ValueError):
                best_preco = 0.0

    if best_match and best_score >= 0.45:
        return best_match, best_preco
    return None


def extract_address(text: str) -> str | None:
    text_lower = text.lower().strip()

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

    text_clean = re.sub(r',\s*$', '', text_clean.strip())
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    patterns = [
        r'(rua|av|avenida|alameda|travessa|estrada)\s+\w+.{3,}',
        r'(rua|av|avenida|r\.)\s+[\w\s]+\s+\d+',
        r'\w[\w\s]+\s+\d+[\w\s]*bairro\s+\w+',
        r'(rua|av)\s+\w+\s+\d+',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            endereco = match.group(0).strip()
            endereco = re.sub(r'[,\s]+$', '', endereco).strip()
            if len(endereco) >= 8:
                return endereco

    if re.search(r'\brua\b', text_clean, re.IGNORECASE) and len(text_clean) >= 8:
        clean = re.sub(r'[,\s]+$', '', text_clean).strip()
        return clean if len(clean) >= 8 else None

    return None


def is_affirmative(text: str) -> bool:
    stripped = text.strip()
    text_norm = unicodedata.normalize('NFD', stripped.lower())
    text_norm = ''.join(c for c in text_norm if unicodedata.category(c) != 'Mn')

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


def is_negative(text: str) -> bool:
    stripped = text.strip()
    if _extract_with_patterns(stripped, NEGATIVE):
        return True
    first = re.split(r'[\s,!.;:]+', stripped.lower(), maxsplit=1)[0]
    return first in {'nao', 'não', 'n', 'negativo', 'errado'}


def is_close_order(text: str) -> bool:
    return _extract_with_patterns(_norm(text), CLOSE_ORDER_PATTERNS)


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


def _adicionar_ao_carrinho(
    session: dict[str, Any],
    tamanhos: list[dict[str, Any]],
    bordas: list[dict[str, Any]],
    cardapio: list[dict[str, Any]],
) -> dict[str, Any]:
    pizza = session.get('pizza_atual') or {}
    sabor = str(pizza.get('sabor') or '').strip()
    tamanho = str(pizza.get('tamanho') or '').strip().upper()
    borda = str(pizza.get('borda') or 'Sem borda').strip() or 'Sem borda'

    sabor_obj = next((s for s in cardapio if str(s.get('nome') or '').strip() == sabor), None)
    tamanho_obj = next((t for t in tamanhos if str(t.get('sigla') or '').strip().upper() == tamanho), None)
    borda_obj = next((b for b in bordas if str(b.get('nome') or '').strip().lower() == borda.lower()), {'preco_extra': 0})

    preco_base = _get_sabor_price(sabor_obj, tamanho, tamanho_obj)
    try:
        preco_borda = float(borda_obj.get('preco_extra') or 0)
    except (TypeError, ValueError):
        preco_borda = 0.0

    preco_total = preco_base + preco_borda

    session['carrinho'].append(
        {
            'sabor': sabor,
            'tamanho': tamanho,
            'borda': borda,
            'preco': preco_total,
        }
    )
    session['pizza_atual'] = {}
    return session


def _resumo_carrinho(carrinho: list[dict[str, Any]]) -> str:
    linhas: list[str] = []
    for p in carrinho:
        sabor = str(p.get('sabor') or '').strip()
        tamanho = str(p.get('tamanho') or '').strip()
        borda = str(p.get('borda') or 'Sem borda').strip()
        try:
            preco = float(p.get('preco') or 0)
        except (TypeError, ValueError):
            preco = 0.0

        borda_str = f' + {borda}' if borda and borda.lower() != 'sem borda' else ''
        linhas.append(f'🍕 {sabor} {tamanho}{borda_str} — R${preco:.2f}'.replace('.', ','))
    return '\n'.join(linhas)


def resumo_pizzas_com_pergunta(session: dict[str, Any]) -> str:
    carrinho = session.get('carrinho', [])
    if not carrinho:
        return 'Mais alguma pizza ou pode fechar? 🍕'

    linhas = ['🛒 *No seu pedido até agora:*\n']
    total = 0.0
    for item in carrinho:
        sabor = str(item.get('sabor') or '').strip()
        tamanho = str(item.get('tamanho') or '').strip()
        borda = str(item.get('borda') or 'Sem borda').strip()
        borda_str = f' + {borda}' if borda and borda.lower() != 'sem borda' else ''
        try:
            preco = float(item.get('preco') or 0)
        except (TypeError, ValueError):
            preco = 0.0
        total += preco
        preco_str = f'R${preco:.2f}'.replace('.', ',')
        linhas.append(f'• {sabor} {tamanho}{borda_str} — {preco_str}'.strip())

    total_str = f'R${total:.2f}'.replace('.', ',')
    linhas.append(f'\n💰 *Total parcial: {total_str}*')
    linhas.append('\nMais alguma pizza ou pode fechar? 🍕')
    return '\n'.join(linhas)


def _montar_resumo(
    session: dict[str, Any],
    tamanhos: list[dict[str, Any]],
    bordas: list[dict[str, Any]],
    cardapio: list[dict[str, Any]],
    tenant_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    _ = (tamanhos, bordas, cardapio)
    carrinho = session.get('carrinho', [])
    bebidas_carrinho = session.get('bebidas_carrinho', [])
    if not isinstance(bebidas_carrinho, list):
        bebidas_carrinho = []

    total = 0.0
    for p in carrinho:
        try:
            total += float(p.get('preco') or 0)
        except (TypeError, ValueError):
            continue

    total_bebidas = 0.0
    for b in bebidas_carrinho:
        try:
            preco = float((b or {}).get('preco') or 0)
        except (TypeError, ValueError):
            preco = 0.0
        try:
            qty = int((b or {}).get('quantidade') or 1)
        except (TypeError, ValueError):
            qty = 1
        if qty <= 0:
            qty = 1
        total_bebidas += preco * qty

    itens: list[dict[str, Any]] = []
    for p in carrinho:
        sabor = str(p.get('sabor') or '').strip()
        tamanho = str(p.get('tamanho') or '').strip()
        borda = str(p.get('borda') or 'Sem borda').strip()
        borda_str = f' + {borda}' if borda and borda.lower() != 'sem borda' else ''
        try:
            preco = float(p.get('preco') or 0)
        except (TypeError, ValueError):
            preco = 0.0
        itens.append(
            {
                'nome': f'{sabor} {tamanho}{borda_str}'.strip(),
                'preco': preco,
                'quantidade': 1,
            }
        )

    for b in bebidas_carrinho:
        nome = str((b or {}).get('nome') or '').strip()
        if not nome:
            continue
        try:
            preco = float((b or {}).get('preco') or 0)
        except (TypeError, ValueError):
            preco = 0.0
        try:
            qty = int((b or {}).get('quantidade') or 1)
        except (TypeError, ValueError):
            qty = 1
        if qty <= 0:
            qty = 1
        itens.append({'nome': nome, 'preco': preco, 'quantidade': qty})

    nome_negocio = str((tenant_config or {}).get('nome_negocio') or 'Pizzaria').strip() or 'Pizzaria'
    total = total + total_bebidas

    session['state'] = PizzaState.FINALIZADO.value
    session['total'] = total

    comprovante = gerar_comprovante(
        itens=itens,
        total=total,
        endereco=str(session.get('endereco', '') or ''),
        pagamento=str(session.get('pagamento', '') or ''),
        nome_negocio=nome_negocio,
        sub_nicho='pizzaria',
    )

    return comprovante, session


def _tamanhos_pergunta(tamanhos: list[dict[str, Any]], sabor_obj: dict[str, Any] | None = None) -> str:
    _EMOJIS = {"P": "🔹", "M": "🔸", "G": "🔶", "GG": "🔥"}
    if tamanhos:
        linhas = []
        for t in tamanhos:
            sigla = str(t.get('sigla') or '').strip().upper()
            if not sigla:
                continue
            nome = str(t.get('nome') or sigla).strip()
            fatias = int(t.get('fatias') or 0)
            fatias_str = f" — {fatias} fatias" if fatias else ""
            emoji = _EMOJIS.get(sigla, "•")
            preco_str = ""
            if sabor_obj:
                preco = _get_sabor_price(sabor_obj, sigla, t)
                if preco:
                    preco_str = f" — R${preco:.2f}".replace(".", ",")
            label = f"{nome} ({sigla})" if nome.upper() != sigla else sigla
            linhas.append(f"{emoji} {label}{fatias_str}{preco_str}")
        return "Qual tamanho? 🍕\n\n" + "\n".join(linhas)
    return "Qual tamanho? 🍕\n\n🔹 Pequena (P) — 4 fatias\n🔸 Média (M) — 6 fatias\n🔶 Grande (G) — 8 fatias\n🔥 GG — 12 fatias"


PAGAMENTO_PIZZA_MSG = (
    "Como vai pagar? 💳\n\n"
    "📚 Pix\n"
    "💵 Dinheiro\n"
    "💳 Cartão"
)


def _tamanhos_curto(tamanhos: list[dict[str, Any]]) -> str:
    if tamanhos:
        values = [str(t.get('sigla') or '').strip().upper() for t in tamanhos if str(t.get('sigla') or '').strip()]
        if values:
            return ' | '.join(values)
    return 'P | M | G | GG'


def _bordas_pergunta(bordas: list[dict[str, Any]]) -> str:
    linhas = []
    for b in bordas:
        nome = str(b.get('nome') or '').strip()
        if not nome or nome.lower() == 'sem borda':
            continue
        preco_extra = float(b.get('preco_extra') or 0)
        preco_str = f" — +R${preco_extra:.2f}".replace(".", ",") if preco_extra else ""
        linhas.append(f"🧀 {nome}{preco_str}")
    linhas.append("❌ Sem borda")
    return "Qual borda? 🧀\n\n" + "\n".join(linhas)


def _detect_inline_checkout_info(text: str, session: dict[str, Any]) -> None:
    if not session.get('endereco'):
        endereco = extract_address(text)
        if endereco:
            session['endereco'] = endereco
    if not session.get('pagamento'):
        pagamento = extract_payment(text)
        if pagamento:
            session['pagamento'] = pagamento


def _detect_inline_pizza_info(text: str, session: dict[str, Any], bordas: list[dict[str, Any]]) -> None:
    pizza_atual = session.get('pizza_atual') or {}
    if not isinstance(pizza_atual, dict) or not pizza_atual.get('sabor'):
        return

    if not pizza_atual.get('tamanho'):
        tamanho = extract_tamanho(text)
        if tamanho:
            pizza_atual['tamanho'] = tamanho[0]

    if not pizza_atual.get('borda'):
        borda = extract_borda(text, bordas)
        if borda:
            pizza_atual['borda'] = borda

    session['pizza_atual'] = pizza_atual


def _apos_adicionar_pizza(
    session: dict[str, Any],
    tamanhos: list[dict[str, Any]],
) -> tuple[str | list[str], dict[str, Any]]:
    fila = session.get('fila_sabores', [])
    if fila:
        proximo = fila.pop(0)
        session['fila_sabores'] = fila
        session['pizza_atual'] = _new_pizza(proximo)
        session['state'] = PizzaState.AGUARDANDO_TAMANHO.value
        return f"E a de {proximo}? Qual tamanho? {_tamanhos_curto(tamanhos)}", session

    ultimo_item = (session.get('carrinho') or [])[-1] if session.get('carrinho') else {}
    sabor = str(ultimo_item.get('sabor') or '').strip()
    tamanho = str(ultimo_item.get('tamanho') or '').strip()
    borda = str(ultimo_item.get('borda') or 'Sem borda').strip()
    try:
        preco = float(ultimo_item.get('preco') or 0)
    except (TypeError, ValueError):
        preco = 0.0
    preco_fmt = f'R${preco:.2f}'.replace('.', ',')

    msg1_linhas = [
        '✅ Anotado!',
        f'🍕 *{sabor} {tamanho}*'.strip(),
    ]
    if borda and borda.lower() != 'sem borda':
        msg1_linhas.append(f'   🧀 {borda}')
    msg1_linhas.append(f'💰 *{preco_fmt}*')

    session['state'] = PizzaState.MAIS_PIZZAS.value
    msg1 = '\n'.join(msg1_linhas)
    msg2 = resumo_pizzas_com_pergunta(session)
    return [msg1, msg2], session


def process_pizza_message(
    text: str,
    session: dict[str, Any],
    cardapio: list[dict[str, Any]],
    tamanhos: list[dict[str, Any]],
    bordas: list[dict[str, Any]],
    bebidas: list[dict[str, Any]],
    tenant_config: dict[str, Any],
) -> tuple[str | list[str], dict[str, Any]]:
    _ = tenant_config

    session = _ensure_session_defaults(session)
    state = session.get('state', PizzaState.AGUARDANDO_PEDIDO.value)
    mensagem = (text or '').strip()

    _detect_inline_checkout_info(mensagem, session)
    _detect_inline_pizza_info(mensagem, session, bordas)

    logger.info(f"[PIZZA-FLOW] state atual: {state}")

    if state == PizzaState.AGUARDANDO_PEDIDO.value:
        sabores = extract_todos_sabores(mensagem, cardapio)
        if not sabores:
            produto_consultado = str(session.get('produto_consultado') or '').strip()
            referencia_vaga = re.search(
                r'\b(quero uma|quero um|me manda|pode ser|entao|então|vou querer|vou de|me ve|me vê)\b',
                _norm(mensagem),
            )
            if produto_consultado and referencia_vaga:
                sabores = [produto_consultado]
                logger.info('[PIZZA] Usando produto consultado: %s', produto_consultado)
        if not sabores:
            return 'Qual sabor de pizza você quer? 🍕', session

        primeiro = sabores[0]
        fila = sabores[1:]

        session['pizza_atual'] = _new_pizza(primeiro)
        session['fila_sabores'] = fila
        session.pop('produto_consultado', None)
        _detect_inline_pizza_info(mensagem, session, bordas)
        session['state'] = PizzaState.CONFIRMANDO_SABOR.value

        return f'Pizza de {primeiro}, certo? 🍕', session

    if state == PizzaState.CONFIRMANDO_SABOR.value:
        if is_negative(mensagem):
            session['pizza_atual'] = {}
            session['state'] = PizzaState.AGUARDANDO_PEDIDO.value
            return 'Tudo bem! Qual sabor você quer então? 🍕', session

        pizza_atual = session.get('pizza_atual') or {}
        if not pizza_atual.get('tamanho'):
            tamanho_extraido = extract_tamanho(mensagem)
            if tamanho_extraido:
                pizza_atual['tamanho'] = tamanho_extraido[0]
        if not pizza_atual.get('borda'):
            borda_extraida = extract_borda(mensagem, bordas)
            if borda_extraida:
                pizza_atual['borda'] = borda_extraida
        session['pizza_atual'] = pizza_atual

        # Se o cliente já informou tamanho e borda junto da confirmação, seguir fluxo sem exigir "sim".
        if pizza_atual.get('tamanho') and pizza_atual.get('borda'):
            _adicionar_ao_carrinho(session, tamanhos, bordas, cardapio)
            return _apos_adicionar_pizza(session, tamanhos)

        if pizza_atual.get('tamanho') and re.search(r'\b(sem borda|borda)\b', _norm(mensagem)):
            session['state'] = PizzaState.AGUARDANDO_BORDA.value
            return _bordas_pergunta(bordas), session

        if is_affirmative(mensagem):
            pizza_atual = session.get('pizza_atual') or {}

            if pizza_atual.get('tamanho') and pizza_atual.get('borda'):
                _adicionar_ao_carrinho(session, tamanhos, bordas, cardapio)
                return _apos_adicionar_pizza(session, tamanhos)

            if pizza_atual.get('tamanho'):
                session['state'] = PizzaState.AGUARDANDO_BORDA.value
                return _bordas_pergunta(bordas), session

            session['state'] = PizzaState.AGUARDANDO_TAMANHO.value
            sabor_obj_conf = next((s for s in cardapio if str(s.get('nome') or '').strip().lower() == str((session.get('pizza_atual') or {}).get('sabor') or '').strip().lower()), None)
            return _tamanhos_pergunta(tamanhos, sabor_obj_conf), session

        novo_sabor = extract_sabor(mensagem, cardapio)
        if novo_sabor:
            session['pizza_atual'] = _new_pizza(novo_sabor[0])
            session['state'] = PizzaState.CONFIRMANDO_SABOR.value
            return f'Pizza de {novo_sabor[0]}, certo? 🍕', session

        sabor = str((session.get('pizza_atual') or {}).get('sabor') or '').strip()
        return f'Confirma: pizza de {sabor}? 🍕', session

    if state == PizzaState.AGUARDANDO_TAMANHO.value:
        pizza_atual = session.get('pizza_atual') or {}
        tamanho_extraido = extract_tamanho(mensagem)
        if tamanho_extraido:
            pizza_atual['tamanho'] = tamanho_extraido[0]
            session['pizza_atual'] = pizza_atual

        if pizza_atual.get('tamanho') and pizza_atual.get('borda'):
            _adicionar_ao_carrinho(session, tamanhos, bordas, cardapio)
            return _apos_adicionar_pizza(session, tamanhos)

        if pizza_atual.get('tamanho'):
            session['state'] = PizzaState.AGUARDANDO_BORDA.value
            return _bordas_pergunta(bordas), session

        sabor_obj_tam = next((s for s in cardapio if str(s.get('nome') or '').strip().lower() == str((session.get('pizza_atual') or {}).get('sabor') or '').strip().lower()), None)
        return _tamanhos_pergunta(tamanhos, sabor_obj_tam), session

    if state == PizzaState.AGUARDANDO_BORDA.value:
        pizza_atual = session.get('pizza_atual') or {}
        borda = extract_borda(mensagem, bordas)
        if borda:
            pizza_atual['borda'] = borda
            session['pizza_atual'] = pizza_atual

        if not pizza_atual.get('borda'):
            return _bordas_pergunta(bordas), session

        _adicionar_ao_carrinho(session, tamanhos, bordas, cardapio)
        return _apos_adicionar_pizza(session, tamanhos)

    if state == PizzaState.MAIS_PIZZAS.value:
        fila = session.get('fila_sabores', [])
        if fila:
            proximo = fila.pop(0)
            session['fila_sabores'] = fila
            session['pizza_atual'] = _new_pizza(proximo)
            session['state'] = PizzaState.AGUARDANDO_TAMANHO.value
            return f"E a de {proximo}? Qual tamanho? {_tamanhos_curto(tamanhos)}", session

        logger.info(f"[MAIS_PIZZAS] texto: '{mensagem}'")
        bebida_result = extract_bebida(mensagem, bebidas)
        logger.info(f"[MAIS_PIZZAS] bebida encontrada: {bebida_result}")
        sabor_result = extract_sabor(mensagem, cardapio)
        logger.info(f"[MAIS_PIZZAS] sabor encontrado: {sabor_result}")

        texto_norm = _norm(mensagem.strip())
        negacao_simples = re.search(r'^(nao|n|nope)[\s!.]*$', texto_norm)

        if bebida_result and not sabor_result:
            session['bebida_sugerida'] = bebida_result[0]
            session['state'] = PizzaState.CONFIRMANDO_BEBIDA.value
            preco_str = f'R${bebida_result[1]:.2f}'.replace('.', ',')
            return f'{bebida_result[0]} — {preco_str}, certo? 🥤', session

        if is_close_order(mensagem) or negacao_simples:
            bebidas_disponiveis = [b for b in bebidas if b.get('disponivel', True)]
            if bebidas_disponiveis:
                exemplos = bebidas_disponiveis[:3]
                exemplos_str = ', '.join([str(b.get('nome') or '').strip() for b in exemplos if str(b.get('nome') or '').strip()])
                sufixo = ' e mais' if len(bebidas_disponiveis) > 3 else ''
                session['state'] = PizzaState.OFERECENDO_BEBIDAS.value
                lista_bebidas = '\n'.join(
                    f"• {str(b.get('nome') or '').strip()} — R${float(b.get('preco') or 0):.2f}".replace('.', ',')
                    for b in bebidas_disponiveis
                    if str(b.get('nome') or '').strip()
                )
                return (
                    '🥤 Vai querer uma bebida?\n\n'
                    f'{lista_bebidas}\n\n'
                    'Ou digita _"só pizzas"_ pra fechar direto!'
                ), session

            if session.get('endereco'):
                if session.get('pagamento'):
                    return _montar_resumo(session, tamanhos, bordas, cardapio, tenant_config)
                session['state'] = PizzaState.AGUARDANDO_PAGAMENTO.value
                return PAGAMENTO_PIZZA_MSG, session

            session['state'] = PizzaState.AGUARDANDO_ENDERECO.value
            return 'Endereço pra entrega? 📍', session

        if sabor_result:
            pizza_nova = _new_pizza(sabor_result[0])
            tamanho_inline = extract_tamanho(mensagem)
            if tamanho_inline:
                pizza_nova['tamanho'] = tamanho_inline[0]
            borda_inline = extract_borda(mensagem, bordas)
            if borda_inline:
                pizza_nova['borda'] = borda_inline
            session['pizza_atual'] = pizza_nova
            session['state'] = PizzaState.CONFIRMANDO_SABOR.value
            return f"Pizza de {sabor_result[0]}, certo? 🍕", session

        if session.get('endereco') and not session.get('pagamento'):
            session['state'] = PizzaState.AGUARDANDO_PAGAMENTO.value
            return PAGAMENTO_PIZZA_MSG, session

        return resumo_pizzas_com_pergunta(session), session

    if state == PizzaState.OFERECENDO_BEBIDAS.value:
        # Permite o cliente seguir direto com endereco/pagamento sem repetir "so isso".
        if session.get('endereco'):
            if session.get('pagamento'):
                return _montar_resumo(session, tamanhos, bordas, cardapio, tenant_config)
            session['state'] = PizzaState.AGUARDANDO_PAGAMENTO.value
            return PAGAMENTO_PIZZA_MSG, session

        for pattern in RECUSA_BEBIDA_PATTERNS:
            if re.search(pattern, _norm(mensagem), re.IGNORECASE):
                session['state'] = PizzaState.AGUARDANDO_ENDERECO.value
                return 'Endereço pra entrega? 📍', session

        bebida = extract_bebida(mensagem, bebidas)
        if bebida:
            session['bebida_sugerida'] = bebida[0]
            session['state'] = PizzaState.CONFIRMANDO_BEBIDA.value
            preco_str = f'R${bebida[1]:.2f}'.replace('.', ',')
            return f'{bebida[0]} — {preco_str}, certo? 🥤', session

        lista = '\n'.join(
            [
                f"• {str(b.get('nome') or '').strip()} — R${float(b.get('preco') or 0):.2f}".replace('.', ',')
                for b in bebidas
                if b.get('disponivel', True)
            ]
        )
        return f'Temos essas bebidas:\n{lista}\n\nQual você quer?', session

    if state == PizzaState.CONFIRMANDO_BEBIDA.value:
        if is_affirmative(mensagem):
            bebida_sugerida = str(session.get('bebida_sugerida') or '').strip()
            bebida_obj = next((b for b in bebidas if str(b.get('nome') or '').strip() == bebida_sugerida), None)
            if bebida_obj:
                qty = extract_quantidade(mensagem)
                if qty <= 0:
                    qty = 1
                if 'bebidas_carrinho' not in session or not isinstance(session.get('bebidas_carrinho'), list):
                    session['bebidas_carrinho'] = []
                session['bebidas_carrinho'].append(
                    {
                        'nome': str(bebida_obj.get('nome') or '').strip(),
                        'preco': float(bebida_obj.get('preco') or 0),
                        'quantidade': qty,
                    }
                )

            session['state'] = PizzaState.OFERECENDO_BEBIDAS.value
            session.pop('bebida_sugerida', None)
            return 'Mais alguma bebida? Ou digita _"só isso"_ pra fechar! 🥤', session

        if is_negative(mensagem):
            session['state'] = PizzaState.AGUARDANDO_ENDERECO.value
            return 'Endereço pra entrega? 📍', session

        nova_bebida = extract_bebida(mensagem, bebidas)
        if nova_bebida:
            session['bebida_sugerida'] = nova_bebida[0]
            preco_str = f'R${nova_bebida[1]:.2f}'.replace('.', ',')
            return f'{nova_bebida[0]} — {preco_str}, certo? 🥤', session

        return f"Confirma: {session.get('bebida_sugerida')}? 🥤", session

    if state == PizzaState.AGUARDANDO_ENDERECO.value:
        if not session.get('endereco'):
            endereco = extract_address(mensagem)
            if not endereco:
                return 'Me manda o endereço completo pra entrega 📍', session
            session['endereco'] = endereco

        if session.get('pagamento'):
            return _montar_resumo(session, tamanhos, bordas, cardapio, tenant_config)

        session['state'] = PizzaState.AGUARDANDO_PAGAMENTO.value
        return PAGAMENTO_PIZZA_MSG, session

    if state == PizzaState.AGUARDANDO_PAGAMENTO.value:
        pagamento = extract_payment(mensagem)
        if not pagamento and not session.get('pagamento'):
            return PAGAMENTO_PIZZA_MSG, session

        if pagamento:
            session['pagamento'] = pagamento

        if not session.get('endereco'):
            session['state'] = PizzaState.AGUARDANDO_ENDERECO.value
            return 'Endereço pra entrega? 📍', session

        return _montar_resumo(session, tamanhos, bordas, cardapio, tenant_config)

    if state == PizzaState.FINALIZADO.value and is_close_order(mensagem):
        return 'Pedido já fechado por aqui. Se quiser outro, me fala o sabor 🍕', session

    return 'Pode mandar seu pedido! 🍕', session
