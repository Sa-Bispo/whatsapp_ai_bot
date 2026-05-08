# query_handler.py — handler de consultas via IA
# Ativa quando cliente faz pergunta sobre produto fora do fluxo de pedido

import asyncio
import logging
import os
import re
from typing import Any

from google import genai

logger = logging.getLogger(__name__)
MODEL_NAME = 'gemini-2.0-flash-lite'

# ---------------------------------------------------------------------------
# Padrões de detecção
# ---------------------------------------------------------------------------

QUERY_PATTERNS = [
    # Disponibilidade
    r"\b(tem|vocês têm|voces tem|tem disponível|tem disponivel|"
    r"ainda tem|tem pra comprar|tem pra vender)\b",
    # Preço
    r"\b(quanto (custa|fica|é|ta|tá)|qual o (preço|valor|preco)|"
    r"quanto (cobram|custam)|preço de|valor do|valor da)\b",
    # Ingredientes / composição
    r"\b(o que (tem|leva|vem)|quais (ingredientes|sabores|opções|opcoes)|"
    r"como (é|vem|é feito)|tem (glúten|gluten|lactose|carne))\b",
    # Comparação
    r"\b(qual (é melhor|vale mais|você indica|me indica)|"
    r"diferença entre|me recomenda)\b",
    # Cardápio geral
    r"\b(o que (vocês têm|voces tem|tem disponível|tem no cardápio)|"
    r"quais (são os|sao os)|me fala (do|da|sobre))\b",
]

ORDER_RESUME_PATTERNS = [
    r"\b(quero|vou (querer|pedir|levar)|me (dá|da|manda|vê|ve)|"
    r"pode (ser|colocar)|coloca|traz|me (anota|separa))\b",
    r"\b(sim|isso|pode ser|vou (ficar|levar) com)\b",
]


def is_product_query(text: str) -> bool:
    """Detecta se a mensagem é uma consulta sobre produto."""
    text_lower = text.lower()
    for pattern in QUERY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Handler principal
# ---------------------------------------------------------------------------

def _extract_consulted_product(text: str, sub_nicho: str, estoque: dict) -> str | None:
    if sub_nicho == 'pizzaria':
        try:
            from pizza_flow import extract_sabor

            sabores = estoque.get('sabores', []) if isinstance(estoque, dict) else []
            resultado = extract_sabor(text, sabores)
            if resultado:
                return resultado[0]
        except Exception:
            return None

    if sub_nicho in {'adega', 'lanchonete'}:
        try:
            if sub_nicho == 'lanchonete':
                from lanchonete_flow import extract_produto
            else:
                from adega_flow import extract_produto

            itens = estoque.get('items', []) if isinstance(estoque, dict) else []
            resultado = extract_produto(text, itens)
            if resultado:
                return resultado[0]
        except Exception:
            return None

    return None


def _fmt_preco(valor: float) -> str:
    return f'R${valor:.2f}'.replace('.', ',')


def _fallback_sem_ia(text: str, sub_nicho: str, estoque: dict, produto_contexto: str | None = None) -> str:
    text_lower = (text or '').lower()

    if sub_nicho == 'adega':
        itens = estoque.get('items', []) if isinstance(estoque, dict) else []
        try:
            from adega_flow import extract_produto

            resultado = extract_produto(text_lower, itens)
            nome = resultado[0] if resultado else (str(produto_contexto or '').strip() or None)
            if nome:
                item = next((i for i in itens if str(i.get('nome', '')).strip() == nome), None)
                if item:
                    preco = float(item.get('preco', item.get('preco_base', 0)) or 0)
                    disponivel = bool(item.get('disponivel', True))
                    status = 'disponivel' if disponivel else 'indisponivel no momento'
                    return f'{nome} ta {_fmt_preco(preco)} e esta {status}. Quer pedir?'
        except Exception:
            pass
        return 'Pode me dizer qual produto voce quer saber?'

    if sub_nicho == 'pizzaria':
        sabores = estoque.get('sabores', []) if isinstance(estoque, dict) else []
        try:
            from pizza_flow import extract_sabor

            resultado = extract_sabor(text_lower, sabores)
            if resultado:
                nome = resultado[0]
                sabor = next((s for s in sabores if str(s.get('nome', '')).strip() == nome), None)
                if sabor:
                    precos = sabor.get('precos', {})
                    precos_str = ' | '.join([f'{k}: {_fmt_preco(float(v))}' for k, v in precos.items()])
                    return f'{nome}: {precos_str}. Quer pedir?'
        except Exception:
            pass
        return 'Pode me dizer qual sabor voce quer saber?'

    if sub_nicho == 'lanchonete':
        itens = estoque.get('items', []) if isinstance(estoque, dict) else []
        try:
            from lanchonete_flow import extract_produto

            # Consulta "tem X" com item fora do cardápio não deve virar match aproximado indevido.
            match_tem = re.search(r'\btem\s+(.+)$', text_lower)
            if match_tem:
                solicitado = re.sub(r'[^a-z0-9]+', ' ', match_tem.group(1).lower()).strip()
                solicitado = re.sub(r'\b(ai|aqui|pra|para|disponivel|disponível)\b', ' ', solicitado).strip()
                if solicitado:
                    nomes_norm = [re.sub(r'[^a-z0-9]+', ' ', str(i.get('nome', '')).lower()).strip() for i in itens]
                    existe_direto = any(
                        solicitado in nome or nome in solicitado
                        for nome in nomes_norm
                        if nome
                    )
                    if not existe_direto:
                        return 'No momento não temos esse item no cardápio.'

            resultado = extract_produto(text_lower, itens)
            nome = resultado[0] if resultado else (str(produto_contexto or '').strip() or None)
            if nome:
                item = next((i for i in itens if str(i.get('nome', '')).strip() == nome), None)
                if item:
                    if re.search(r'\btem\b', text_lower):
                        prefixo = 'Sim, temos. '
                    else:
                        prefixo = ''
                    if item.get('tem_variacoes') and item.get('variacoes'):
                        variacoes = item.get('variacoes') or []
                        resumo = ' | '.join([
                            f"{str(v.get('sigla') or '').upper()}: {_fmt_preco(float(v.get('preco') or 0))}"
                            for v in variacoes
                        ])
                        return f'{prefixo}{nome} tem tamanhos: {resumo}. Quer pedir qual?'
                    preco = float(item.get('preco', item.get('preco_base', 0)) or 0)
                    return f'{prefixo}{nome} ta {_fmt_preco(preco)}. Quer pedir?'
        except Exception:
            pass
        return 'Pode me dizer qual item voce quer saber?'

    return 'Pode me dizer o que voce quer saber?'


async def handle_product_query(
    text: str,
    tenant_id: str,
    sub_nicho: str,
    estoque: dict,
    tenant_config: dict,
    gemini_model: Any | None = None,
    produto_contexto: str | None = None,
) -> tuple[str, dict]:
    """Retorna (resposta, contexto)."""
    nome_negocio = tenant_config.get("nome_negocio", "nossa loja")
    nome_atendente = tenant_config.get("nome_atendente", "Atendente")

    # Montar contexto de estoque por sub-nicho
    if sub_nicho == "adega":
        itens = estoque.get("items", [])
        contexto_estoque = "\n".join([
            f"- {i['nome']}: R${float(i.get('preco', 0)):.2f} "
            f"({'disponível' if i.get('disponivel', True) else 'indisponível'})"
            for i in itens
        ])
        tipo_negocio = "adega de bairro"

    elif sub_nicho == "lanchonete":
        itens = estoque.get("items", [])
        linhas = []
        for i in itens:
            if i.get('tem_variacoes') and i.get('variacoes'):
                variacoes = " | ".join([
                    f"{v.get('sigla', '')}: R${float(v.get('preco', 0)):.2f}"
                    for v in i.get('variacoes', [])
                ])
                linhas.append(
                    f"- {i['nome']}: {variacoes} "
                    f"({'disponível' if i.get('disponivel', True) else 'indisponível'})"
                )
            else:
                linhas.append(
                    f"- {i['nome']}: R${float(i.get('preco', 0)):.2f} "
                    f"({'disponível' if i.get('disponivel', True) else 'indisponível'})"
                )
        contexto_estoque = "\n".join(linhas)
        tipo_negocio = "lanchonete"

    elif sub_nicho == "pizzaria":
        sabores = estoque.get("sabores", [])
        tamanhos = estoque.get("tamanhos", [])
        bordas = estoque.get("bordas", [])
        bebidas = estoque.get("bebidas", [])

        linhas_sabores = []
        for s in sabores:
            if not s.get("disponivel", True):
                continue
            precos = s.get("precos", {})
            precos_str = " | ".join([f"{k}: R${float(v):.2f}" for k, v in precos.items()])
            linhas_sabores.append(f"- {s['nome']}: {precos_str}")

        linhas_tamanhos = [
            f"- {t['sigla']} ({t['nome']}): {t['fatias']} fatias"
            for t in tamanhos
        ]
        linhas_bordas = [
            f"- {b['nome']}: +R${float(b['preco_extra']):.2f}"
            for b in bordas
        ]
        linhas_bebidas = [
            f"- {b['nome']}: R${float(b['preco']):.2f}"
            for b in bebidas if b.get("ativo", True)
        ]

        contexto_estoque = (
            "SABORES DISPONÍVEIS E PREÇOS:\n" + "\n".join(linhas_sabores) +
            "\n\nTAMANHOS:\n" + "\n".join(linhas_tamanhos) +
            "\n\nBORDAS:\n" + "\n".join(linhas_bordas) +
            ("\n\nBEBIDAS:\n" + "\n".join(linhas_bebidas) if linhas_bebidas else "")
        )
        tipo_negocio = "pizzaria"

    else:
        contexto_estoque = "Estoque não disponível."
        tipo_negocio = "loja"

    system_prompt = f"""Você é {nome_atendente}, atendente virtual da {nome_negocio}, uma {tipo_negocio}.

ESTOQUE ATUAL:
{contexto_estoque}

REGRAS OBRIGATÓRIAS:
1. Responda APENAS sobre produtos do estoque acima. Nunca invente produtos.
2. Se o produto não estiver no estoque, diga que não temos disponível.
3. Máximo 3 linhas por resposta. Seja direto e objetivo.
4. Tom: descontraído e simpático, como um atendente de balcão.
5. Se o cliente parecer interessado em comprar, termine com uma pergunta que convide ao pedido.
   Ex: "Quer pedir um?" / "Mando um pra você?" / "Vou separar?"
6. NUNCA pergunte endereço, pagamento ou quantidade aqui — isso é do fluxo de pedido.
7. Responda em português brasileiro informal.

Pergunta do cliente: {text}"""

    produto_consultado = _extract_consulted_product(text, sub_nicho, estoque)
    if not produto_consultado and produto_contexto:
        produto_consultado = str(produto_contexto).strip() or None
    contexto = {'ultimo_produto_consultado': produto_consultado}
    prompt_completo = f'{system_prompt}\n\nPergunta do cliente: {text}'

    # Para adega/lanchonete com item identificado, responder via fallback estruturado primeiro.
    if sub_nicho in {'adega', 'lanchonete'} and produto_consultado:
        return _fallback_sem_ia(text, sub_nicho, estoque, produto_contexto=produto_consultado), contexto

    try:
        if gemini_model is not None:
            response = await asyncio.wait_for(
                gemini_model.generate_content_async(
                    contents=prompt_completo,
                ),
                timeout=8.0,
            )
            resposta = (response.text or '').strip()
            logger.info('[QUERY] IA respondeu consulta (modelo injetado): %s', resposta[:80])
            return resposta, contexto

        api_key = (os.getenv('GEMINI_API_KEY') or '').strip()
        if not api_key:
            logger.error('[QUERY] GEMINI_API_KEY ausente - usando fallback sem IA')
            return _fallback_sem_ia(text, sub_nicho, estoque, produto_contexto=produto_consultado), contexto

        configured_model = (os.getenv('GEMINI_MODEL_NAME') or '').strip()
        fast_model = (os.getenv('GEMINI_FAST_MODEL') or MODEL_NAME).strip()
        model_candidates = [m for m in [fast_model, configured_model, 'gemini-1.5-flash'] if m]

        seen: set[str] = set()
        unique_candidates: list[str] = []
        for model_name in model_candidates:
            if model_name not in seen:
                seen.add(model_name)
                unique_candidates.append(model_name)
        unique_candidates = unique_candidates[:1]

        client = genai.Client(api_key=api_key)
        last_error = None

        for candidate in unique_candidates:
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=candidate,
                        contents=prompt_completo,
                    ),
                    timeout=1.0,
                )
                resposta = (response.text or '').strip()
                logger.info('[QUERY] IA respondeu consulta (%s): %s', candidate, resposta[:80])
                return resposta, contexto
            except asyncio.TimeoutError:
                logger.warning('[QUERY] Timeout com modelo %s', candidate)
                last_error = 'timeout'
            except Exception as exc:
                logger.warning('[QUERY] Falha com modelo %s: %s', candidate, exc)
                last_error = exc

        logger.error('[QUERY] Falha em todos os modelos: %s', last_error)
        return _fallback_sem_ia(text, sub_nicho, estoque, produto_contexto=produto_consultado), contexto
    except Exception as exc:
        logger.error('[QUERY] Erro na IA: %s', exc)
        return _fallback_sem_ia(text, sub_nicho, estoque, produto_contexto=produto_consultado), contexto
