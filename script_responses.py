"""
script_responses.py — Respostas de script parametrizadas por tenant e sub-nicho.
Cada função monta a resposta usando dados da config e estoque, economizando tokens.
"""

from typing import Optional, Dict, Any
from collections import defaultdict
import random


def resposta_saudacao(tenant_config: dict) -> str:
    """Retorna boas-vindas curtas e variáveis por sub-nicho."""
    nome = tenant_config.get("nome_negocio", "nossa loja")
    atendente = tenant_config.get("nome_atendente", "pessoal")
    sub_nicho = tenant_config.get("sub_nicho", "adega")

    templates = {
        "adega": [
            f"Eee, salve! 🍺 Aqui é a *{nome}*, pode pedir!",
            f"Oi! Bem-vindo à *{nome}* 🍺 O que vai ser hoje?",
            f"Salve! Aqui é {atendente} da *{nome}*. Manda o pedido!",
        ],
        "lanchonete": [
            f"Olá! Bem-vindo à *{nome}* 🥪 O que vai ser hoje?",
            f"Oi! Aqui é {atendente} da *{nome}*. Me fala o pedido!",
        ],
        "pizzaria": [
            f"Oi! Aqui é {atendente} da *{nome}* 🍕 Qual vai ser a pizza hoje?",
            f"Salve! *{nome}* na área 🍕 Me fala o sabor!",
        ],
    }

    opcoes = templates.get(sub_nicho, [f"Olá! Bem-vindo à *{nome}*! O que posso ajudar?"])
    return random.choice(opcoes)


def resposta_horario(tenant_config: dict) -> str:
    """Retorna horários de funcionamento formatados."""
    nome = tenant_config.get("nome_negocio", "nossa loja")
    turnos = tenant_config.get("horarios", [])

    if not turnos:
        return (
            f"🕐 Horário de funcionamento do {nome} ainda não foi configurado.\n"
            f"Entre em contato para mais informações!"
        )

    dias_map = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"]
    linhas = [f"🕐 *Horários do {nome}:*\n"]

    for turno in turnos:
        if not turno.get("ativo", True):
            continue

        dias_nums = turno.get("dias", [])
        dias = [dias_map[d % 7] for d in dias_nums if 0 <= d < 7]
        dias_str = ", ".join(dias) if dias else "Não configurado"

        abertura = turno.get("abertura", "00:00")
        fechamento = turno.get("fechamento", "00:00")
        nome_turno = turno.get("nome", "Funcionamento")

        linhas.append(f"*{nome_turno}*: {dias_str}")
        linhas.append(f"⏰ {abertura} às {fechamento}\n")

    return "".join(linhas).strip()


def resposta_fora_horario(tenant_config: dict) -> str:
    """Resposta quando está fora do horário."""
    nome = tenant_config.get("nome_negocio", "nossa loja")
    turnos = tenant_config.get("horarios", [])
    proximo = ""

    if turnos:
        primeiro = next((t for t in turnos if t.get("ativo", True)), None)
        if primeiro:
            abertura = primeiro.get("abertura", "")
            if abertura:
                proximo = f" Voltamos às *{abertura}*."

    return (
        f"😴 Estamos fechados no momento.{proximo}\n\n"
        f"Pode mandar sua mensagem que respondemos assim que abrirmos! ✉️"
    )


def resposta_entrega(tenant_config: dict) -> str:
    """Retorna modalidades de entrega/retirada."""
    delivery = tenant_config.get("faz_delivery", False)
    retirada = tenant_config.get("faz_retirada", False)
    tempo = tenant_config.get("tempo_entrega", "30-40 minutos")

    linhas = ["🛵 *Modalidades de atendimento:*\n"]

    if delivery:
        linhas.append(f"✅ *Delivery* — entregamos na sua casa")
        linhas.append(f"   Tempo estimado: {tempo}\n")
    if retirada:
        linhas.append("✅ *Retirada no balcão* — você vem buscar\n")
    if not delivery and not retirada:
        linhas.append("Entre em contato para mais informações.")

    return "".join(linhas).strip()


def resposta_status_pedido(ultimo_pedido: Optional[dict]) -> str:
    """Retorna status do último pedido."""
    if not ultimo_pedido:
        return (
            "📦 Não encontrei nenhum pedido recente seu.\n"
            "Quer fazer um pedido agora? 😊"
        )

    status_map = {
        "NOVO": "🟡 recebido e aguardando confirmação",
        "PREPARANDO": "🔵 sendo preparado agora",
        "ENVIADO": "🟢 saiu para entrega",
        "CONCLUIDO": "✅ entregue",
        "FINALIZADO": "✅ entregue",
        "CANCELADO": "❌ cancelado",
    }

    status = ultimo_pedido.get("status", "NOVO")
    status_text = status_map.get(status, status)
    itens = ultimo_pedido.get("itens_resumo", "seu pedido")

    return f"📦 *Status do seu pedido:*\n{itens}\n\nSituação: {status_text}"


def resposta_cardapio(tenant_config: dict, estoque: dict) -> str:
    """Retorna cardápio formatado por sub-nicho."""
    sub_nicho = tenant_config.get("sub_nicho", "adega")

    if sub_nicho == "adega":
        return _cardapio_adega(tenant_config, estoque)
    elif sub_nicho == "lanchonete":
        return _cardapio_lanchonete(tenant_config, estoque)
    elif sub_nicho == "pizzaria":
        return _cardapio_pizzaria(tenant_config, estoque)

    return "Nosso cardápio está sendo atualizado. Pergunte o que você quer! 😊"


def _cardapio_adega(tenant_config: dict, estoque: dict) -> str:
    """Cardápio para adega."""
    nome = tenant_config.get("nome_negocio", "nossa loja")
    itens = [i for i in estoque.get("items", []) if i.get("disponivel")]
    combos = [c for c in estoque.get("combos", []) if c.get("disponivel")]

    if not itens and not combos:
        return (
            f"🍺 Nosso cardápio está sendo atualizado.\n"
            f"Fala o que você quer! 🍺"
        )

    por_cat = defaultdict(list)
    for item in itens:
        cat = item.get("categoria", "Outros")
        por_cat[cat].append(item)

    linhas = [f"🍺 *Cardápio — {nome}*\n"]

    for cat in sorted(por_cat.keys()):
        produtos = por_cat[cat]
        linhas.append(f"*{cat}*")
        for p in produtos:
            preco = f"R${p['preco']:.2f}".replace(".", ",")
            linhas.append(f"• {p['nome']} — {preco}")
        linhas.append("")

    if combos:
        linhas.append("*🔥 Combos*")
        for c in combos:
            preco = f"R${c['preco']:.2f}".replace(".", ",")
            linhas.append(f"• {c['nome']} — {preco}")
            if c.get("descricao"):
                linhas.append(f"  _{c['descricao']}_")
        linhas.append("")

    linhas.append("O que você vai querer? 😄")
    return "\n".join(linhas)


def _cardapio_lanchonete(tenant_config: dict, estoque: dict) -> str:
    """Cardápio para lanchonete."""
    nome = tenant_config.get("nome_negocio", "nossa loja")
    itens = [i for i in estoque.get("items", []) if i.get("disponivel")]
    combos = [c for c in estoque.get("combos", []) if c.get("disponivel")]

    if not itens and not combos:
        return (
            f"🥪 Nosso cardápio está sendo atualizado.\n"
            f"Fala o que você quer! 🥪"
        )

    por_cat = defaultdict(list)
    for item in itens:
        cat = item.get("categoria", "Outros")
        por_cat[cat].append(item)

    linhas = [f"🥪 *Cardápio — {nome}*\n"]

    for cat in sorted(por_cat.keys()):
        produtos = por_cat[cat]
        linhas.append(f"*{cat}*")
        for p in produtos:
            preco = f"R${p['preco']:.2f}".replace(".", ",")
            linhas.append(f"• {p['nome']} — {preco}")
        linhas.append("")

    if combos:
        linhas.append("*🔥 Combos do dia*")
        for c in combos:
            preco = f"R${c['preco']:.2f}".replace(".", ",")
            linhas.append(f"• {c['nome']} — {preco}")
            if c.get("descricao"):
                linhas.append(f"  _{c['descricao']}_")
        linhas.append("")

    linhas.append("O que vai ser hoje? 😊")
    return "\n".join(linhas)


def _cardapio_pizzaria(tenant_config: dict, estoque: dict) -> str:
    """Cardápio para pizzaria."""
    nome = tenant_config.get("nome_negocio", "nossa loja")
    sabores = [s for s in estoque.get("sabores", []) if s.get("disponivel")]
    tamanhos = estoque.get("tamanhos", [])
    bordas = estoque.get("bordas", [])

    if not sabores:
        return (
            f"🍕 Nosso cardápio está sendo atualizado.\n"
            f"Fala o que você quer! 🍕"
        )

    por_cat = defaultdict(list)
    for s in sabores:
        cat = s.get("categoria", "Outros")
        por_cat[cat].append(s)

    linhas = [f"🍕 *Cardápio — {nome}*\n"]

    for cat in sorted(por_cat.keys()):
        lista = por_cat[cat]
        linhas.append(f"*{cat}*")
        for s in lista:
            precos = s.get("precos", {})
            if precos:
                preco_str = " | ".join(
                    [f"{k}:R${v:.0f}" for k, v in sorted(precos.items())]
                )
            else:
                preco_str = "consulte valores"
            linhas.append(f"• {s['nome']} — {preco_str}")
        linhas.append("")

    if tamanhos:
        linhas.append("*Tamanhos disponíveis*")
        for t in tamanhos:
            nome_t = t.get("nome", t.get("sigla", ""))
            fatias = t.get("fatias", 0)
            extra = t.get("modificador_preco", 0)
            extra_str = f"+R${extra:.0f}" if extra > 0 else "base"
            linhas.append(f"• {nome_t} ({fatias} fatias) — {extra_str}")
        linhas.append("")

    if bordas:
        linhas.append("*Bordas disponíveis*")
        for b in bordas:
            extra = b.get("preco_extra", 0)
            extra_str = f"+R${extra:.2f}" if extra > 0 else "sem acréscimo"
            linhas.append(f"• {b['nome']} — {extra_str}")
        linhas.append("")

    linhas.append("Me fala o sabor e o tamanho! 🍕")
    return "\n".join(linhas)


def resposta_cancelamento_confirmacao(tenant_config: dict) -> str:
    """Solicita confirmação de cancelamento."""
    return (
        "Tem certeza que deseja cancelar seu pedido?\n\n"
        "Digite *SIM* para confirmar o cancelamento ou *NÃO* para manter."
    )
