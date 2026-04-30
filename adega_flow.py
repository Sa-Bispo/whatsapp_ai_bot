from __future__ import annotations

import logging
import re
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


def _fmt_brl(valor: float) -> str:
    return f'R${float(valor):,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


class AdegaState(Enum):
    AGUARDANDO_PEDIDO = 'aguardando_pedido'
    CONFIRMANDO_PRODUTO = 'confirmando_produto'
    MAIS_ITENS = 'mais_itens'
    AGUARDANDO_ENDERECO = 'aguardando_endereco'
    AGUARDANDO_PAGAMENTO = 'aguardando_pagamento'
    FINALIZADO = 'finalizado'


STOPWORDS = [
    r'^(opa|oi|ola|ol[aá]|bom dia|boa tarde|boa noite)[,\s]+',
    r'\b(tem|voc[êe]s t[êe]m|voces tem|tem dispon[ií]vel|tem disponivel)\s+',
    r'\b(me ve|me v[eê]|me da|me d[aá]|quero|queria|pode ser|vou querer|traz|coloca)\s+',
    r'\b(a[ií]|la|l[aá]|por favor|pf|pfv|pra mim)\b',
    r'\b(uma|um|umas|uns)\s+',
]

PAYMENT_MAP = {
    'pix': 'Pix',
    'dinheiro': 'Dinheiro',
    'dinheirinho': 'Dinheiro',
    'cartao': 'Cartao',
    'cartão': 'Cartao',
    'debito': 'Debito',
    'débito': 'Debito',
    'credito': 'Credito',
    'crédito': 'Credito',
    'maquina': 'Cartao',
    'máquina': 'Cartao',
}

CLOSE_ORDER_PATTERNS = [
    r'\b(s[óo] isso|so isso|[ée] isso|e isso|pode fechar|fecha|'
    r's[óo] isso mesmo|so isso mesmo|mais n[aã]o|mais nao|'
    r'[ée] tudo|e tudo|pode|quero mais nada|nao quero mais|'
    r'n[aã]o quero mais|mais nada|nada mais|t[aá] bom|ta bom)\b',
]

CONSULTA_PATTERNS = [
    r'^(e\s+\w+[\w\s]*\?)$',
    r'^(\w+[\w\s]*)\?$',
    r'\b(tem|voc[êe]s t[êe]m|existe|tem dispon[ií]vel)\b.*\?',
    r'^(e\s+)?\w+[\w\s]{1,20}\?$',
]

RESUMO_CARRINHO_PATTERNS = [
    r'\b(quais|qual|o que|que) (itens?|produtos?|coisas?|tenho|esta|estao|tem|temos|ficou|ficaram) (no carrinho|anotado|no pedido|ate agora)\b',
    r'\b(me (mostra|fala|diz|conta)) (o que|quais|os itens)\b',
    r'\b(resumo|lista) (do pedido|do carrinho)\b',
    r'\b(o que (tem|esta|temos|ficou)) (no pedido|no carrinho|anotado)\b',
]

AFFIRMATIVE = [
    r'^(sim|s|isso|pode ser|certo|exato|confirmo|pode|ok|yes)[\s!.]*$',
    r'\b(sim|confirmo|pode|isso|certo)\b',
    r'(ja falei|j[aá] falei|falei que sim|[ée] sim|claro que sim)',
]

NEGATIVE = [
    r'^(n[aã]o|nao|n|negativo|errado|outro|diferente)[\s!.]*$',
]

ADDRESS_NOISE = [
    r'\b(pix|dinheiro|cart[aã]o|debito|d[eé]bito|credito|cr[eé]dito|maquina|m[aá]quina)\b',
    r'\b(s[óo] isso|so isso|pode fechar|[ée] isso)\b',
    r'\b(vou pagar (no|na|com)|pagando (no|na|com))\b',
    r'\b(no pix|na maquina|no dinheiro|no cartao|no cart[aã]o)\b',
]


def _norm(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', (text or '').lower())
        if unicodedata.category(c) != 'Mn'
    )


def extract_produto(text: str, estoque: list[dict]) -> Optional[tuple[str, float]]:
    text_lower = (text or '').lower().strip()
    text_clean = text_lower
    for sw in STOPWORDS:
        text_clean = re.sub(sw, '', text_clean, flags=re.IGNORECASE).strip()

    best_match = None
    best_score = 0.0
    for item in estoque:
        if not item.get('disponivel', True):
            continue
        nome = str(item.get('nome') or '').strip()
        if not nome:
            continue
        score = max(
            token_set_ratio(text_lower, nome.lower()) / 100,
            token_set_ratio(text_clean, nome.lower()) / 100,
        )
        if score > best_score:
            best_score = score
            best_match = nome

    if best_score >= 0.45:
        return best_match, best_score
    return None


def extract_quantidade(text: str) -> int:
    word_to_num = {
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
    text_norm = _norm(text)
    for word, num in word_to_num.items():
        if re.search(rf'\b{word}\b', text_norm):
            return num

    patterns = [
        r'(?:sim|pode|certo|isso|quero)[,\s]+(?:quero\s+)?(\d+)',
        r'(?:quero|me\s+manda|me\s+da|me\s+ve|coloca)\s+(\d+)',
        r'\b(\d+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_norm)
        if match:
            try:
                return max(1, int(match.group(1)))
            except ValueError:
                pass
    return 1


def extract_pagamento(text: str) -> Optional[str]:
    text_norm = _norm(text)
    for key, val in PAYMENT_MAP.items():
        if re.search(rf'\b{_norm(key)}\b', text_norm):
            return val
    return None


def extract_endereco(text: str) -> Optional[str]:
    text_lower = (text or '').lower().strip()
    text_clean = text_lower
    for pattern in ADDRESS_NOISE:
        text_clean = re.sub(pattern, '', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r',\s*$', '', text_clean.strip())
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    patterns = [
        r'(rua|av|avenida|alameda|travessa|estrada)\s+\w+.{3,}',
        r'(rua|av|avenida|r\.)\s+[\w\s]+\s+\d+',
        r'\w[\w\s]+\s+\d+[\w\s]*bairro\s+\w+',
        r'(rua|av)\s+\w+\s+\d+',
        # Tolerantes: exigem indicador explícito de número de endereço
        r'.{3,}\s+(n[°º.]?|n[uú]mero)\s*\d+',
        r'\w[\w\s]{3,},\s*(n[°º.]?|n[uú]mero)\s*\d+',
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
    text_norm = _norm((text or '').strip())
    return any(re.search(p, text_norm, re.IGNORECASE) for p in AFFIRMATIVE)


def is_negative(text: str) -> bool:
    text_norm = _norm((text or '').strip())
    return any(re.search(p, text_norm, re.IGNORECASE) for p in NEGATIVE)


def is_close_order(text: str) -> bool:
    text_norm = _norm(text)
    return any(re.search(p, text_norm, re.IGNORECASE) for p in CLOSE_ORDER_PATTERNS)


def is_consulta_produto(text: str) -> bool:
    text_lower = (text or '').lower().strip()
    for pattern in CONSULTA_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def is_pedindo_resumo(text: str) -> bool:
    text_norm = _norm(text)
    for pattern in RESUMO_CARRINHO_PATTERNS:
        if re.search(pattern, text_norm, re.IGNORECASE):
            return True
    return False


def _is_more_items_prompt(text: str) -> bool:
    text_norm = _norm(text)
    return bool(re.search(r'\b(mais alguma coisa|mais algo|mais alguma|algo mais)\b', text_norm))


def has_price_question(text: str) -> bool:
    return bool(
        re.search(
            r'(quanto(s)?\s+(fica|vai|custa|[ée]|ta)|qual\s+o\s+(preco|valor)|quanto\s+(fica|[ée]))',
            (text or '').lower(),
        )
    )


def _get_item(estoque: list[dict], nome: str) -> dict | None:
    return next((i for i in estoque if str(i.get('nome', '')).strip() == nome), None)


def _get_preco(estoque: list[dict], nome: str) -> float:
    item = _get_item(estoque, nome)
    if not item:
        return 0.0
    return float(item.get('preco', item.get('preco_base', 0)) or 0)


def _add_item_to_cart(session: dict, nome: str, quantidade: int, preco: float) -> None:
    carrinho = session.get('carrinho', [])
    carrinho.append({
        'nome': nome,
        'quantidade': int(quantidade),
        'preco': float(preco),
        '_confirmado': True,
    })
    session['carrinho'] = carrinho


def _garantir_item_atual_no_carrinho(session: dict, estoque: list[dict]) -> None:
    """Garante que o produto atual da session esta no carrinho antes de avancar."""
    if session.get('_item_atual_no_carrinho'):
        return

    produto = str(session.get('produto') or '').strip()
    if not produto:
        return

    qty = int(session.get('quantidade', 1) or 1)
    preco = _get_preco(estoque, produto)
    _add_item_to_cart(session, produto, qty, preco)
    session['_item_atual_no_carrinho'] = True
    logger.info('[ADEGA] Item adicionado ao carrinho: %s x%s', produto, qty)


def extract_multiplos_itens(text: str, estoque: list[dict]) -> list[dict]:
    """
    Extrai multiplos pares (produto, quantidade) da mesma mensagem.
    Ex: "3 absolut e 4 smirnoff" -> [{"nome": "Absolut", "qty": 3}, {"nome": "Smirnoff", "qty": 4}]
    """
    word_to_num = {
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

    text_norm = _norm(text)
    for word, num in word_to_num.items():
        text_norm = re.sub(rf'\b{word}\b', str(num), text_norm)

    partes = re.split(r'\s*(?:,|\b(?:e|mais)\b)\s*', text_norm, flags=re.IGNORECASE)

    resultados: list[dict] = []
    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue

        qty = 1
        match_qty = re.search(r'\b(\d+)\b', parte)
        if match_qty:
            qty = int(match_qty.group(1))
            parte_sem_num = re.sub(r'\b\d+\b', '', parte).strip()
        else:
            parte_sem_num = parte

        parte_sem_num = re.sub(
            r'^(manda|manda ai|manda aí|coloca|traz|quero|queria|me manda|me da|me dá)\s+',
            '',
            parte_sem_num,
            flags=re.IGNORECASE,
        ).strip()

        if len(parte_sem_num) < 2:
            continue

        resultado = extract_produto(parte_sem_num, estoque)
        if resultado:
            resultados.append({'nome': resultado[0], 'qty': qty})

    return resultados


def montar_resumo_parcial(session: dict, estoque: list[dict]) -> str:
    """Monta resumo do carrinho atual com precos."""
    _garantir_item_atual_no_carrinho(session, estoque)
    carrinho = session.get('carrinho', [])

    if not carrinho:
        produto = session.get('produto')
        if produto:
            item_obj = next((i for i in estoque if i['nome'] == produto), None)
            preco = float(item_obj.get('preco', item_obj.get('preco_base', 0)) or 0) if item_obj else 0
            qty = int(session.get('quantidade', 1) or 1)
            total = preco * qty
            return (
                f'No carrinho agora:\n'
                f'• {qty}x {produto} — { _fmt_brl(total) }\n\n'
                f'Total: { _fmt_brl(total) }\n\n'
                'Mais alguma coisa ou pode fechar?'
            )
        return 'Carrinho vazio ainda. O que voce quer pedir? 🍺'

    linhas: list[str] = []
    total = 0.0
    for item in carrinho:
        if item.get('_confirmado') is False:
            continue
        qty = int(item.get('quantidade', 1) or 1)
        preco = float(item.get('preco', 0) or 0)
        subtotal = preco * qty
        total += subtotal
        linhas.append(f'• {qty}x {item.get("nome", "")} — {_fmt_brl(subtotal)}')

    total_fmt = _fmt_brl(total)
    itens_str = '\n'.join(linhas)

    return (
        f'No carrinho agora:\n{itens_str}\n\n'
        f'💰 Total ate agora: {total_fmt}\n\n'
        'Mais alguma coisa ou pode fechar? 🍺'
    )


def _cart_total(carrinho: list[dict]) -> float:
    total = 0.0
    for item in carrinho:
        total += float(item.get('preco', 0) or 0) * int(item.get('quantidade', 1) or 1)
    return total


def process_adega_message(
    text: str,
    session: dict,
    estoque: list[dict],
    tenant_config: dict,
) -> tuple[Optional[str], dict]:
    """
    Processa mensagem no fluxo de adega.
    Retorna (resposta, session) e resposta None para fallback de IA.
    """
    state = session.get('state', AdegaState.AGUARDANDO_PEDIDO.value)

    if not session.get('pagamento'):
        pag = extract_pagamento(text)
        if pag:
            session['pagamento'] = pag

    if not session.get('endereco'):
        end = extract_endereco(text)
        if end:
            session['endereco'] = end

    if state == AdegaState.AGUARDANDO_PEDIDO.value:
        if is_consulta_produto(text):
            return None, session

        multiplos = extract_multiplos_itens(text, estoque)
        if len(multiplos) > 1:
            session['fila_itens'] = multiplos[1:]
            primeiro = multiplos[0]
            session['produto_sugerido'] = primeiro['nome']
            session['quantidade'] = primeiro['qty']
            session.pop('produto_consultado', None)
            session['state'] = AdegaState.CONFIRMANDO_PRODUTO.value
            qty_str = f"{primeiro['qty']}x " if int(primeiro['qty']) > 1 else ''
            return f'{qty_str}{primeiro["nome"]}, certo? 🍺', session

        resultado = extract_produto(text, estoque)

        # Se vier referencia vaga (ex: "quero 12 dela"), usar o ultimo produto consultado.
        if not resultado:
            produto_consultado = str(session.get('produto_consultado') or '').strip()
            referencias_vagas = re.search(
                r'\b(quero|vou querer|pode ser|me manda|me da|me dá|'
                r'traz|coloca|dela|dele|disso|desse|dessa|um|uma)\b',
                (text or '').lower(),
            )
            if produto_consultado and referencias_vagas:
                resolvido = extract_produto(produto_consultado, estoque)
                nome_resolvido = resolvido[0] if resolvido else produto_consultado
                resultado = (nome_resolvido, 0.9)
                logger.info('[ADEGA] Usando produto consultado: %s -> %s', produto_consultado, nome_resolvido)

        if resultado:
            nome, _score = resultado
            qty = extract_quantidade(text)
            session['produto_sugerido'] = nome
            session['quantidade'] = qty
            session.pop('produto_consultado', None)
            session['state'] = AdegaState.CONFIRMANDO_PRODUTO.value
            qty_str = f'{qty}x ' if qty > 1 else ''
            return f'{qty_str}{nome}, certo? 🍺', session
        return None, session

    if state == AdegaState.CONFIRMANDO_PRODUTO.value:
        if is_pedindo_resumo(text):
            return montar_resumo_parcial(session, estoque), session

        if is_affirmative(text):
            session['produto'] = session.get('produto_sugerido')
            qty = extract_quantidade(text)
            if qty > 1:
                session['quantidade'] = qty
            session['_item_atual_no_carrinho'] = False
            _garantir_item_atual_no_carrinho(session, estoque)

            fila = session.get('fila_itens', [])
            if fila:
                proximo = fila.pop(0)
                session['fila_itens'] = fila
                session['produto_sugerido'] = proximo['nome']
                session['quantidade'] = proximo['qty']
                qty_str = f"{proximo['qty']}x " if int(proximo['qty']) > 1 else ''
                return f'E {qty_str}{proximo["nome"]}, certo? 🍺', session

            session.pop('fila_itens', None)

            prefixo = ''
            if has_price_question(text):
                nome_prod = str(session.get('produto') or '').strip()
                preco = _get_preco(estoque, nome_prod)
                q = int(session.get('quantidade', 1) or 1)
                total = preco * q
                if q > 1:
                    prefixo = f'Fica {_fmt_brl(total)} ({q}x {_fmt_brl(preco)}). '
                else:
                    prefixo = f'Fica {_fmt_brl(preco)}. '

            session['state'] = AdegaState.MAIS_ITENS.value
            return f'{prefixo}Mais alguma coisa? Ou pode fechar! 🍺', session

        if is_negative(text):
            novo = extract_produto(text, estoque)
            if novo:
                session['produto_sugerido'] = novo[0]
                session['quantidade'] = extract_quantidade(text)
                qty = int(session.get('quantidade', 1) or 1)
                qty_str = f'{qty}x ' if qty > 1 else ''
                return f'{qty_str}{novo[0]}, certo? 🍺', session
            session.pop('produto_sugerido', None)
            session['state'] = AdegaState.AGUARDANDO_PEDIDO.value
            return 'Tudo bem! Qual produto voce quer? 🍺', session

        novo = extract_produto(text, estoque)
        if novo:
            session['produto_sugerido'] = novo[0]
            session['quantidade'] = extract_quantidade(text)
            qty = int(session.get('quantidade', 1) or 1)
            qty_str = f'{qty}x ' if qty > 1 else ''
            return f'{qty_str}{novo[0]}, certo? 🍺', session

        qty = extract_quantidade(text)
        if qty > 1:
            session['quantidade'] = qty
            return f'{qty}x {session.get("produto_sugerido", "")}, certo? 🍺', session
        return f'Confirma: {session.get("produto_sugerido", "")}? 🍺', session

    if state == AdegaState.MAIS_ITENS.value:
        if is_pedindo_resumo(text):
            return montar_resumo_parcial(session, estoque), session

        if session.get('aguardando_confirmacao_multiplos'):
            if is_affirmative(text):
                session.pop('aguardando_confirmacao_multiplos', None)
                return 'Mais alguma coisa? Ou pode fechar! 🍺', session
            if is_negative(text):
                session.pop('aguardando_confirmacao_multiplos', None)
                return 'Beleza! Me fala os itens novamente do jeito que voce preferir. 🍺', session

        if _is_more_items_prompt(text):
            return 'Mais alguma coisa? Ou pode fechar! 🍺', session

        if is_consulta_produto(text):
            return None, session

        multiplos = extract_multiplos_itens(text, estoque)
        if len(multiplos) > 1 and not is_close_order(text):
            _garantir_item_atual_no_carrinho(session, estoque)
            if 'carrinho' not in session:
                session['carrinho'] = []

            novos_nomes: list[str] = []
            for item in multiplos:
                item_obj = _get_item(estoque, item['nome'])
                preco = float(item_obj.get('preco', item_obj.get('preco_base', 0)) or 0) if item_obj else 0.0
                session['carrinho'].append({
                    'nome': item['nome'],
                    'quantidade': int(item['qty']),
                    'preco': preco,
                    '_confirmado': True,
                })
                qty_str = f"{item['qty']}x " if int(item['qty']) > 1 else ''
                novos_nomes.append(f"{qty_str}{item['nome']}")

            session['aguardando_confirmacao_multiplos'] = True
            itens_str = ' + '.join(novos_nomes)
            return f'{itens_str}, certo? 🍺', session

        novo = extract_produto(text, estoque)
        if novo and not is_close_order(text):
            _garantir_item_atual_no_carrinho(session, estoque)
            session['produto_sugerido'] = novo[0]
            session['quantidade'] = extract_quantidade(text)
            session['state'] = AdegaState.CONFIRMANDO_PRODUTO.value
            qty = int(session.get('quantidade', 1) or 1)
            qty_str = f'{qty}x ' if qty > 1 else ''
            return f'{qty_str}{novo[0]}, certo? 🍺', session

        if is_close_order(text) or is_negative(text) or re.search(r'^(n[aã]o|nao|n)[\s!.]*$', _norm(text)):
            _garantir_item_atual_no_carrinho(session, estoque)
            if session.get('endereco') and session.get('pagamento'):
                return _montar_resumo(session, estoque, tenant_config)
            if session.get('endereco'):
                session['state'] = AdegaState.AGUARDANDO_PAGAMENTO.value
                return 'Pix, dinheiro ou cartao? 💳', session
            session['state'] = AdegaState.AGUARDANDO_ENDERECO.value
            return 'Endereco pra entrega? 📍', session

        # Pergunta aberta não reconhecida → cai na IA, mantém estado
        return None, session

    if state == AdegaState.AGUARDANDO_ENDERECO.value:
        if session.get('endereco'):
            if session.get('pagamento'):
                return _montar_resumo(session, estoque, tenant_config)
            session['state'] = AdegaState.AGUARDANDO_PAGAMENTO.value
            return 'Pix, dinheiro ou cartao? 💳', session
        return 'Me manda o endereco completo pra entrega 📍', session

    if state == AdegaState.AGUARDANDO_PAGAMENTO.value:
        if session.get('pagamento'):
            return _montar_resumo(session, estoque, tenant_config)
        return 'Qual a forma de pagamento? Pix, dinheiro ou cartao 💳', session

    return None, session


def _montar_resumo(session: dict, estoque: list[dict], tenant_config: dict) -> tuple[str, dict]:
    _garantir_item_atual_no_carrinho(session, estoque)
    carrinho = session.get('carrinho', [])

    total = _cart_total(carrinho)
    nome_negocio = str(tenant_config.get('nome_negocio') or 'Adega').strip() or 'Adega'

    session['state'] = AdegaState.FINALIZADO.value
    session['total'] = total

    itens_receipt = [
        {'nome': i['nome'], 'preco': float(i['preco']), 'quantidade': int(i['quantidade'])}
        for i in carrinho
    ]

    comprovante = gerar_comprovante(
        itens=itens_receipt,
        total=total,
        endereco=session.get('endereco', ''),
        pagamento=session.get('pagamento', ''),
        nome_negocio=nome_negocio,
        sub_nicho='adega',
    )
    return comprovante, session


def save_adega_order_payload(session: dict) -> dict:
    carrinho = session.get('carrinho', [])
    produto = str(session.get('produto') or '').strip()
    quantidade = int(session.get('quantidade', 1) or 1)

    if not carrinho and produto:
        carrinho = [{
            'nome': produto,
            'quantidade': quantidade,
            'preco': float(session.get('total', 0) or 0) / max(quantidade, 1),
        }]

    total = float(session.get('total', 0) or 0)
    if total <= 0:
        total = _cart_total(carrinho)

    return {
        'carrinho': carrinho,
        'total': total,
        'endereco': session.get('endereco', ''),
        'pagamento': session.get('pagamento', ''),
        'produto': produto,
        'quantidade': quantidade,
    }
