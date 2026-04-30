from __future__ import annotations

EMOJIS = {
    'adega': '🍺',
    'lanchonete': '🥪',
    'pizzaria': '🍕',
}

PREVISAO = {
    'adega': '30-40 minutos',
    'lanchonete': '20-30 minutos',
    'pizzaria': '40-50 minutos',
}


def _alinhar_item(nome: str, preco: float, largura: int = 32) -> str:
    preco_str = f'R${preco:.2f}'.replace('.', ',')
    espacos = largura - len(nome) - len(preco_str)
    pontos = '.' * max(espacos, 1)
    return f'• {nome} {pontos} {preco_str}'


def gerar_comprovante(
    itens: list[dict],
    total: float,
    endereco: str,
    pagamento: str,
    nome_negocio: str,
    sub_nicho: str,
    numero_pedido: int | None = None,
) -> str:
    emoji = EMOJIS.get(sub_nicho, '🛒')
    previsao = PREVISAO.get(sub_nicho, '30-40 minutos')
    total_str = f'R${total:.2f}'.replace('.', ',')
    numero_str = f" #{'%04d' % numero_pedido}" if numero_pedido else ''

    linhas_itens: list[str] = []
    for item in itens:
        nome = str(item.get('nome', ''))
        preco = float(item.get('preco', 0.0) or 0.0)
        qty = int(item.get('quantidade', 1) or 1)
        if qty > 1:
            nome = f'{qty}x {nome}'
        linhas_itens.append(_alinhar_item(nome, preco * qty))
    itens_str = '\n'.join(linhas_itens)

    comprovante = (
        '╔══════════════════╗\n'
        f'✅ PEDIDO CONFIRMADO{numero_str}\n'
        '╚══════════════════╝\n\n'
        f'{emoji} *{nome_negocio}*\n\n'
        '🛒 *Seu pedido:*\n'
        f'{itens_str}\n\n'
        f'💰 *Total: {total_str}*\n'
        f'💳 Pagamento: {pagamento}\n\n'
        '📍 *Entrega em:*\n'
        f'{endereco}\n\n'
        f'⏱ Previsão: {previsao}\n\n'
        '_Guarde esse comprovante._\n'
        '_Dúvidas? É só chamar aqui!_ 😊'
    )
    return comprovante
