"""
router.py — Roteador de intenção pré-IA para interceptar mensagens antes de chamar Gemini.
Detecta gatilhos de script com regex e retorna respostas predefinidas economizando tokens.
"""

import re
from datetime import datetime
from typing import Optional

# Gatilhos por intenção — patterns case-insensitive
TRIGGERS = {
    "saudacao": [
        r"^(oi|olá|ola|bom dia|boa tarde|boa noite|hey|hello|e aí|eai|tudo bem)[\s!?.]*$"
    ],
    "horario": [
        r"(hor[áa]rio|que horas|quando abre|quando fecha|aberto|fechado|funcionamento)"
    ],
    "entrega": [
        r"(faz entrega|tem delivery|entrega[sz]?|motoboy|retirada|buscar|pegar)"
    ],
    "status_pedido": [
        r"(status|meu pedido|cadê|onde est[áa]|quanto tempo|previs[ãa]o)"
    ],
    "cardapio": [
        r"(card[áa]pio|o que tem|o que voc[êe]s t[êe]m|ver produtos|mostrar produtos|lista|tem o qu[êe])"
    ],
    "cancelar": [
        r"(cancelar|cancela|quero cancelar|desistir)"
    ],
    "fechar_pedido": [
        r"(só isso|so isso|é isso|e isso|pode fechar|fecha o pedido|só isso mesmo|so isso mesmo|só isso só|so isso so|por enquanto é isso|por enquanto e isso|tá bom assim|ta bom assim|pode ser isso)"
    ],
}


def detect_intent(text: str) -> Optional[str]:
    """
    Detecta intenção da mensagem combinando contra regex patterns.
    Retorna nome da intenção (string) ou None se deve ir pra IA.
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    for intent, patterns in TRIGGERS.items():
        for pattern in patterns:
            try:
                if re.search(pattern, text_lower):
                    return intent
            except Exception:
                # Padrão regex inválido — ignora e continua
                pass

    return None


async def detect_intent_with_context(
    text: str,
    tenant_id: str,
    phone: str,
    sub_nicho: str,
    redis_client
) -> Optional[str]:
    """
    Detecta intenção considerando contexto da sessão.
    Se é pizzaria e tem sessão ativa, não interceptar — deixar máquina de estados tratar.
    """
    if sub_nicho == "pizzaria":
        session_key = f"pizza_session:{tenant_id}:{phone}"
        try:
            session_raw = await redis_client.get(session_key) if redis_client else None
        except Exception:
            session_raw = None
        
        if session_raw:
            # Sessão ativa — não interceptar nenhuma mensagem
            return None
        
        # Sem sessão — só interceptar saudação e horário
        intent = detect_intent(text)
        if intent in ["saudacao", "horario", "status_pedido"]:
            return intent
        # Qualquer outra coisa vai pro fluxo de pizzaria
        return None
    
    # Para outros nichos, roteamento normal
    return detect_intent(text)


def is_within_hours(horarios: list) -> bool:
    """
    Verifica se está dentro do horário de funcionamento.
    horarios: lista de dicts com formato:
    [
        {
            "nome": "Funcionamento",
            "dias": [0, 1, 2, 3, 4, 5, 6],  # 0=dom, 1=seg, ..., 6=sáb
            "abertura_minutos": 600,  # 10:00 = 600 min
            "fechamento_minutos": 1200,  # 20:00 = 1200 min
            "ativo": True
        }
    ]
    """
    if not horarios or not isinstance(horarios, list):
        return True  # Se não há horários configurados, considera sempre aberto

    now = datetime.now()
    current_day = now.weekday() + 1 if now.weekday() < 6 else 0  # Python: 0=seg, 6=dom → nosso: 0=dom
    current_time = now.hour * 60 + now.minute

    for turno in horarios:
        if not turno.get("ativo", True):
            continue

        dias = turno.get("dias", [])
        if current_day not in dias:
            continue

        abertura_minutos = turno.get("abertura_minutos", 0)
        fechamento_minutos = turno.get("fechamento_minutos", 1440)

        if abertura_minutos <= current_time <= fechamento_minutos:
            return True

    return False
