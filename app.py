import asyncio
import json
import logging
import re
import time
import uuid

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from chains import generate_persona_response, _get_redis_sync
from database_api import (
    get_tenant_by_instance,
    get_tenant_configs,
    create_produto,
    fetch_stock_for_context,
    get_ultimo_pedido,
    save_pizza_order,
    save_order,
)
from adega_flow import process_adega_message, AdegaState, save_adega_order_payload
from lanchonete_flow import process_lanchonete_message, LanchoneteState, save_lanchonete_order_payload
from query_handler import is_product_query, handle_product_query
from message_buffer import buffer_message
from pizza_flow import process_pizza_message, PizzaState
from config import GEMINI_API_KEY
from evolution_api import send_whatsapp_message
from router import detect_intent, detect_intent_with_context, is_within_hours
from script_responses import (
    resposta_saudacao,
    resposta_horario,
    resposta_fora_horario,
    resposta_entrega,
    resposta_status_pedido,
    resposta_cardapio,
    resposta_cancelamento_confirmacao,
)
from memory import get_session_history
from order_extractor import build_order_payload_from_history_window_async
from order_extractor import extract_confirmed_product_from_history, extract_quantity_from_confirmation


# Configure logging to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

app = FastAPI()
logger = logging.getLogger(__name__)


def _normalize_lookup(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', (value or '').lower()).strip()


def _extract_address_from_message(message: str) -> str:
    text = (message or '').strip()
    if not text:
        return ''

    pattern = re.compile(r'(?i)(?:moro na\s+)?((?:rua|r\.|avenida|av\.?|travessa|tv\.?|alameda|estrada)\s+.+)')
    match = pattern.search(text)
    if not match:
        return ''

    address = match.group(1).strip(' ,.-')
    address = re.sub(
        r'(?i)(?:[\.,;:\- ]+)?(?:vou pagar|pagamento|pago no|pagar no|pix|cart[aã]o|dinheiro).*$','',
        address,
    ).strip(' ,.-')
    return address


def _normalize_messages(reply_value: str | list[str] | None) -> tuple[list[str], str]:
    if isinstance(reply_value, list):
        mensagens = [str(msg).strip() for msg in reply_value if str(msg).strip()]
        ultimo = mensagens[-1] if mensagens else ''
        return mensagens, ultimo
    texto = str(reply_value or '').strip()
    return ([texto] if texto else []), texto

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:3000',
        'http://127.0.0.1:3000',
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ============================================================================
# Modelos Pydantic para Visão Computacional (Import de Cardápio)
# ============================================================================

class ProdutoVisionResponse(BaseModel):
    """Schema para um produto extraído por visão computacional."""
    nome: str
    preco_base: float
    categoria: str
    regras_ia: str = Field(default='')


class ProdutosVisionList(BaseModel):
    """Array de produtos extraído pela IA."""
    produtos: list[ProdutoVisionResponse]


gemini_client = genai.Client(api_key=(GEMINI_API_KEY or '').strip()) if GEMINI_API_KEY else None


def extract_chat_id(payload: dict) -> str | None:
    raw_chat_id = payload.get('data', {}).get('key', {}).get('remoteJid')

    if not raw_chat_id or raw_chat_id.endswith('@g.us'):
        return None

    number = raw_chat_id.split('@')[0]
    number = re.sub(r'\D', '', number)
    return number or None


def extract_message_text(payload: dict) -> str | None:
    message_data = payload.get('data', {}).get('message', {})

    possible_texts = [
        message_data.get('conversation'),
        message_data.get('extendedTextMessage', {}).get('text'),
        message_data.get('imageMessage', {}).get('caption'),
        message_data.get('videoMessage', {}).get('caption'),
    ]

    for text in possible_texts:
        if isinstance(text, str) and text.strip():
            return text.strip()

    return None


def extract_instance_name(payload: dict) -> str | None:
    instance = payload.get('instance')

    if isinstance(instance, str) and instance.strip():
        return instance.strip()

    if isinstance(instance, dict):
        nested_name = instance.get('instanceName') or instance.get('name')
        if isinstance(nested_name, str) and nested_name.strip():
            return nested_name.strip()

    data_instance = payload.get('data', {}).get('instance')
    if isinstance(data_instance, str) and data_instance.strip():
        return data_instance.strip()

    return None


async def _process_chat_message(
    tenant_id: str,
    message: str,
    phone: str,
    session_id: str | None = None,
):
    """Shared message router used by both API simulator and WhatsApp webhook."""
    t0 = time.time()

    tenant_id = (tenant_id or '').strip()
    message = (message or '').strip()
    phone = (phone or '').strip()
    session_id = (session_id or phone or tenant_id).strip()

    if not tenant_id or not message:
        raise ValueError('tenant_id e message são obrigatórios.')

    try:
        configs = await get_tenant_configs(tenant_id)
        logger.info('[ROUTER] sub_nicho: %s', configs.get('sub_nicho'))
        conversation_id = f'{tenant_id}:{session_id}'
        
        # PARTE 1: ROTEADOR — verificar intenções pré-IA
        horarios = configs.get('horarios', [])
        if horarios and not is_within_hours(horarios):
            reply = resposta_fora_horario(configs)
            return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}

        import redis
        redis_async = redis.asyncio.from_url(
            'redis://localhost:6379/6',
            decode_responses=True
        )
        intent = await detect_intent_with_context(
            text=message,
            tenant_id=tenant_id,
            phone=session_id,
            sub_nicho=str(configs.get('sub_nicho') or '').strip().lower(),
            redis_client=redis_async
        )
        if intent:
            if intent == 'saudacao':
                historico = get_session_history(conversation_id)
                if not historico.messages:
                    reply = resposta_saudacao(configs)
                    return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}
            elif intent == 'horario':
                reply = resposta_horario(configs)
                return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}
            elif intent == 'entrega':
                reply = resposta_entrega(configs)
                return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}
            elif intent == 'status_pedido':
                ultimo_pedido = await asyncio.to_thread(
                    get_ultimo_pedido,
                    phone=session_id,
                    tenant_id=tenant_id,
                )
                reply = resposta_status_pedido(ultimo_pedido)
                return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}
            elif intent == 'cardapio':
                estoque = await fetch_stock_for_context(tenant_id)
                reply = resposta_cardapio(configs, estoque)
                return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}
            elif intent == 'cancelar':
                reply = resposta_cancelamento_confirmacao(configs)
                return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}

        if configs.get('sub_nicho') == 'pizzaria':
            import redis
            redis_client = _get_redis_sync()
            phone = session_id
            session_key = f'pizza_session:{tenant_id}:{phone}'

            try:
                session_raw = redis_client.get(session_key) if redis_client else None
                session = json.loads(session_raw) if session_raw else {}
            except Exception:
                session = {}

            state = str(session.get('state', PizzaState.AGUARDANDO_PEDIDO.value) or '')
            active_states = {
                PizzaState.CONFIRMANDO_SABOR.value,
                PizzaState.AGUARDANDO_TAMANHO.value,
                PizzaState.AGUARDANDO_BORDA.value,
                PizzaState.MAIS_PIZZAS.value,
                PizzaState.OFERECENDO_BEBIDAS.value,
                PizzaState.CONFIRMANDO_BEBIDA.value,
                PizzaState.AGUARDANDO_ENDERECO.value,
                PizzaState.AGUARDANDO_PAGAMENTO.value,
            }

            try:
                estoque = await fetch_stock_for_context(tenant_id)
            except Exception:
                estoque = {}

            if state not in active_states and is_product_query(message):
                resposta_consulta, contexto_consulta = await handle_product_query(
                    text=message,
                    tenant_id=tenant_id,
                    sub_nicho='pizzaria',
                    estoque=estoque if isinstance(estoque, dict) else {},
                    tenant_config=configs,
                )
                ultimo_produto = (contexto_consulta or {}).get('ultimo_produto_consultado')
                if ultimo_produto:
                    session['produto_consultado'] = ultimo_produto
                if session and redis_client:
                    try:
                        redis_client.setex(session_key, 1800, json.dumps(session))
                    except Exception:
                        pass
                return {'reply': resposta_consulta, 'response': resposta_consulta, 'sale_complete': False, 'confetti': False}

            cardapio = estoque.get('sabores', []) if isinstance(estoque, dict) else []
            tamanhos = estoque.get('tamanhos', []) if isinstance(estoque, dict) else []
            bordas = estoque.get('bordas', []) if isinstance(estoque, dict) else []
            bebidas = estoque.get('bebidas', []) if isinstance(estoque, dict) else []

            reply, session_atualizada = process_pizza_message(
                text=message,
                session=session,
                cardapio=cardapio,
                tamanhos=tamanhos,
                bordas=bordas,
                bebidas=bebidas,
                tenant_config=configs,
            )
            mensagens_reply, ultimo_reply = _normalize_messages(reply)

            state_value = (session_atualizada or {}).get('state')
            sale_complete = (
                state_value == PizzaState.FINALIZADO.value
                or any('✅' in msg or 'Pedido confirmado' in msg for msg in mensagens_reply)
            )

            if sale_complete and session_atualizada and not session_atualizada.get('pedido_salvo'):
                try:
                    saved_order = await save_pizza_order(tenant_id=tenant_id, phone=phone, session=session_atualizada)
                    if saved_order:
                        session_atualizada['pedido_salvo'] = True
                        logger.info(
                            '[SAVE_ORDER] Pedido salvo antes do envio (pizzaria): order_id=%s',
                            str((saved_order or {}).get('id') or ''),
                        )
                except Exception as exc:
                    logger.error(f'[CHAT-SYNC] erro ao salvar pedido pizzaria: {exc}', exc_info=True)

            if redis_client and session_atualizada:
                try:
                    redis_client.setex(session_key, 1800, json.dumps(session_atualizada))
                except Exception:
                    pass

            payload = {
                'reply': ultimo_reply,
                'response': ultimo_reply,
                'sale_complete': sale_complete,
                'confetti': sale_complete,
            }
            if len(mensagens_reply) > 1:
                payload['messages'] = mensagens_reply
            return payload

        sub_nicho = str(configs.get('sub_nicho') or '').strip().lower()
        if sub_nicho in ('adega', 'lanchonete'):
            redis_sync = _get_redis_sync()
            phone = session_id
            session_key = f'{sub_nicho}_session:{tenant_id}:{phone}'

            try:
                session_raw = redis_sync.get(session_key) if redis_sync else None
                session = json.loads(session_raw) if session_raw else {}
            except Exception:
                session = {}

            if sub_nicho == 'adega':
                state = str(session.get('state', AdegaState.AGUARDANDO_PEDIDO.value) or '')
                active_states = {
                    AdegaState.CONFIRMANDO_PRODUTO.value,
                    AdegaState.MAIS_ITENS.value,
                    AdegaState.AGUARDANDO_ENDERECO.value,
                    AdegaState.AGUARDANDO_PAGAMENTO.value,
                }
            else:
                state = str(session.get('state', LanchoneteState.AGUARDANDO_PEDIDO.value) or '')
                active_states = {
                    LanchoneteState.CONFIRMANDO_ITEM.value,
                    LanchoneteState.AGUARDANDO_TAMANHO.value,
                    LanchoneteState.AGUARDANDO_ADICIONAIS.value,
                    LanchoneteState.MAIS_ITENS.value,
                    LanchoneteState.AGUARDANDO_ENDERECO.value,
                    LanchoneteState.AGUARDANDO_PAGAMENTO.value,
                }

            try:
                estoque_data = await fetch_stock_for_context(tenant_id)
            except Exception:
                estoque_data = {}

            if state not in active_states and is_product_query(message):
                resposta_consulta, contexto_consulta = await handle_product_query(
                    text=message,
                    tenant_id=tenant_id,
                    sub_nicho=sub_nicho,
                    estoque=estoque_data if isinstance(estoque_data, dict) else {},
                    tenant_config=configs,
                    produto_contexto=str(session.get('produto_consultado') or session.get('produto_sugerido') or '').strip(),
                )

                ultimo_produto = (contexto_consulta or {}).get('ultimo_produto_consultado')
                if not ultimo_produto:
                    itens = estoque_data.get('items', []) if isinstance(estoque_data, dict) else []
                    texto_msg = (message or '').lower()
                    texto_resposta = (resposta_consulta or '').lower()
                    for item in itens:
                        nome_item = str(item.get('nome') or '').strip()
                        if not nome_item:
                            continue
                        nome_lower = nome_item.lower()
                        if nome_lower in texto_msg or nome_lower in texto_resposta:
                            ultimo_produto = nome_item
                            break

                if not ultimo_produto:
                    match_nome = re.search(r'^\s*([^,.!?]+?)\s+ta\s+r\$', (resposta_consulta or '').lower())
                    if match_nome:
                        ultimo_produto = match_nome.group(1).strip().title()

                if ultimo_produto and sub_nicho in ('adega', 'lanchonete'):
                    session['produto_consultado'] = ultimo_produto

                if session and redis_sync:
                    try:
                        redis_sync.setex(session_key, 1800, json.dumps(session, ensure_ascii=False))
                    except Exception:
                        pass
                return {'reply': resposta_consulta, 'response': resposta_consulta, 'sale_complete': False, 'confetti': False}

            estoque = estoque_data.get('items', []) if isinstance(estoque_data, dict) else []

            if sub_nicho == 'adega':
                reply_flow, session_atualizada = process_adega_message(
                    text=message, session=session, estoque=estoque, tenant_config=configs,
                )
                state_finalizado = AdegaState.FINALIZADO.value
            else:
                reply_flow, session_atualizada = process_lanchonete_message(
                    text=message, session=session, estoque=estoque, tenant_config=configs,
                )
                state_finalizado = LanchoneteState.FINALIZADO.value

            if reply_flow is not None:
                mensagens_flow, ultimo_flow = _normalize_messages(reply_flow)
                sale_complete = session_atualizada.get('state') == state_finalizado
                try:
                    if redis_sync:
                        redis_sync.setex(session_key, 1800, json.dumps(session_atualizada, ensure_ascii=False))
                except Exception:
                    pass

                if sale_complete and not session_atualizada.get('pedido_salvo'):
                    save_succeeded = False
                    try:
                        if sub_nicho == 'adega':
                            p = save_adega_order_payload(session_atualizada)
                            cart_items = [
                                {
                                    'product_name': i['nome'],
                                    'name': i['nome'],
                                    'quantity': i['quantidade'],
                                    'price': i['preco'],
                                }
                                for i in p.get('carrinho', [])
                            ]
                            if not cart_items and p.get('produto'):
                                quantidade = max(int(p.get('quantidade', 1) or 1), 1)
                                total = float(p.get('total', 0) or 0)
                                cart_items = [{
                                    'product_name': p['produto'],
                                    'name': p['produto'],
                                    'quantity': quantidade,
                                    'price': total / quantidade if quantidade else 0,
                                }]
                            try:
                                await asyncio.wait_for(
                                    save_order(
                                        tenant_id=tenant_id,
                                        phone=phone,
                                        nome='Cliente WhatsApp',
                                        endereco=p['endereco'],
                                        cart_items=cart_items,
                                        total=p['total'],
                                        forma_pagamento=p['pagamento'],
                                    ),
                                    timeout=5.0,
                                )
                                save_succeeded = True
                                logger.info('[SAVE_ORDER] Pedido salvo antes do envio (adega)')
                                logger.info('[ADEGA-CHAT-SYNC] pedido salvo com sucesso: %s', tenant_id)
                            except asyncio.TimeoutError:
                                logger.error('[ADEGA-CHAT-SYNC] TIMEOUT ao salvar pedido (5s) - tenant: %s', tenant_id)
                            except Exception as save_exc:
                                logger.error('[ADEGA-CHAT-SYNC] erro ao salvar pedido: %s', save_exc, exc_info=True)
                        else:
                            p = save_lanchonete_order_payload(session_atualizada)
                            payload_lanchonete = {
                                'tenant_id': tenant_id,
                                'phone': phone,
                                'nome': 'Cliente WhatsApp',
                                'endereco': p.get('endereco'),
                                'cart_items': [
                                    {
                                        'product_name': i['nome'],
                                        'name': i['nome'],
                                        'quantity': i['quantidade'],
                                        'price': i['preco'],
                                    }
                                    for i in p['carrinho']
                                ],
                                'total': p.get('total'),
                                'forma_pagamento': p.get('pagamento'),
                            }
                            logger.info('[SAVE_ORDER] Tentando salvar pedido lanchonete: %s', payload_lanchonete)
                            try:
                                save_result = await asyncio.wait_for(
                                    save_order(
                                        tenant_id=payload_lanchonete['tenant_id'],
                                        phone=payload_lanchonete['phone'],
                                        nome=payload_lanchonete['nome'],
                                        endereco=payload_lanchonete['endereco'],
                                        cart_items=payload_lanchonete['cart_items'],
                                        total=payload_lanchonete['total'],
                                        forma_pagamento=payload_lanchonete['forma_pagamento'],
                                    ),
                                    timeout=5.0,
                                )
                                save_succeeded = True
                                logger.info(
                                    '[SAVE_ORDER] Pedido lanchonete salvo com sucesso: order_id=%s',
                                    str((save_result or {}).get('id') or ''),
                                )
                                logger.info('[SAVE_ORDER] Pedido salvo antes do envio (lanchonete)')
                                logger.info('[LANCHONETE-CHAT-SYNC] pedido salvo com sucesso: %s', tenant_id)
                            except asyncio.TimeoutError:
                                logger.error('[LANCHONETE-CHAT-SYNC] TIMEOUT ao salvar pedido (5s) - tenant: %s', tenant_id)
                                raise
                            except Exception as save_exc:
                                logger.error('[SAVE_ORDER] FALHA ao salvar pedido lanchonete: %s', save_exc, exc_info=True)
                                raise

                        if save_succeeded:
                            session_atualizada['pedido_salvo'] = True
                            if redis_sync:
                                redis_sync.delete(session_key)
                        else:
                            logger.warning('[CHAT-SYNC] pedido nao foi salvo, mantendo em cache para retry')
                            if redis_sync:
                                redis_sync.setex(session_key, 900, json.dumps(session_atualizada, ensure_ascii=False))
                    except Exception as exc:
                        logger.error('[%s-CHAT-SYNC] erro geral ao salvar pedido: %s', sub_nicho.upper(), exc, exc_info=True)

                payload = {
                    'reply': ultimo_flow,
                    'response': ultimo_flow,
                    'sale_complete': sale_complete,
                    'confetti': sale_complete,
                }
                if len(mensagens_flow) > 1:
                    payload['messages'] = mensagens_flow
                return payload

        prompt_ia = configs.get('promptIa') or ''
        bot_objective = configs.get('botObjective') or 'FECHAR_PEDIDO'

        instruction_ia = 'Responda à mensagem do usuário de forma natural, seguindo sua persona.'
        _mais_itens_states = {'mais_itens'}
        sub_nicho_ia = str(configs.get('sub_nicho') or '').strip().lower()
        if sub_nicho_ia in ('adega', 'lanchonete'):
            _sk = f'{sub_nicho_ia}_session:{tenant_id}:{session_id}'
            try:
                _redis_ai = _get_redis_sync()
                _sr = _redis_ai.get(_sk) if _redis_ai else None
                _sess = json.loads(_sr) if _sr else {}
            except Exception:
                _sess = {}
            if _sess.get('state') in _mais_itens_states:
                _prod = str(_sess.get('produto') or _sess.get('produto_sugerido') or _sess.get('item_sugerido') or '').strip()
                _qty = int(_sess.get('quantidade', 1) or _sess.get('quantidade_sugerida', 1) or 1)
                _carrinho = _sess.get('carrinho', [])
                _carrinho_str = ', '.join(
                    f"{i.get('quantidade', 1)}x {i.get('nome', '')}"
                    for i in _carrinho
                ) if _carrinho else ''
                _ctx_parts = []
                if _prod:
                    _ctx_parts.append(f'Produto atual: {_qty}x {_prod}')
                if _carrinho_str:
                    _ctx_parts.append(f'Itens no carrinho: {_carrinho_str}')
                if _ctx_parts:
                    instruction_ia = (
                        f'CONTEXTO DO PEDIDO ATUAL:\n'
                        + '\n'.join(_ctx_parts)
                        + '\n\nREGRA CRÍTICA: Nunca invente quantidade em estoque. '
                        'Se o cliente perguntar sobre estoque de outros produtos ou quantidades disponíveis, '
                        "diga apenas: 'Não tenho essa informação agora, mas posso anotar seu pedido!' "
                        "Responda a dúvida e termine com: 'Posso fechar o pedido?'"
                    )

        reply: str = await asyncio.to_thread(
            generate_persona_response,
            instruction_ia,
            message,
            conversation_id,
            prompt_ia or None,
            bot_objective,
            tenant_id,
        )
    except Exception as exc:
        print(f'[chat-sync] erro ao gerar resposta: {exc}')
        raise

    ai_response = (reply or '').rstrip()
    is_sale_complete = ai_response.endswith('✅')

    payload = {
        'reply': ai_response,
        'response': ai_response,
        'sale_complete': is_sale_complete,
        'summary': ai_response if is_sale_complete else None,
        'confetti': is_sale_complete,
    }

    logger.info('[CHAT-SYNC] processed in %.2fms tenant=%s session=%s', (time.time() - t0) * 1000, tenant_id, session_id)
    return payload

@app.post('/api/chat-sync')
async def chat_sync(request: Request):
    """Porta síncrona para o simulador da Landing Page.
    Recebe tenant_id + message, carrega a persona do banco e devolve a
    resposta da IA sem disparar nada na Evolution API.
    Inclui roteador de script para intenções pré-IA.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({'error': 'JSON inválido.'}, status_code=400)

    tenant_id = (data.get('tenant_id') or '').strip()
    message = (data.get('message') or '').strip()
    # session_id permite manter histórico separado por visitante.
    # Prioriza session_id explícito, depois phone, e por fim tenant_id.
    phone = (data.get('phone') or '').strip()
    session_id = (data.get('session_id') or phone or tenant_id).strip()

    if not tenant_id or not message:
        return JSONResponse(
            {'error': 'Os campos tenant_id e message são obrigatórios.'},
            status_code=422,
        )

    try:
        payload = await _process_chat_message(
            tenant_id=tenant_id,
            message=message,
            phone=phone,
            session_id=session_id,
        )
    except Exception as exc:
        print(f'[chat-sync] erro ao gerar resposta: {exc}')
        return JSONResponse(
            {'error': 'Erro interno ao processar a mensagem.'},
            status_code=500,
        )

    return payload


@app.post('/webhook')
async def webhook(request: Request):
    data = await request.json()
    event = data.get('event')
    instance_name = extract_instance_name(data)

    if not instance_name:
        return {'status': 'ignored'}

    tenant_id = await get_tenant_by_instance(instance_name)
    if not tenant_id:
        # Ignora mensagens de instâncias não cadastradas no SaaS.
        return {'status': 'ignored'}

    print(f"[WEBHOOK] evento recebido: {event}")
    chat_id = extract_chat_id(data)
    message = extract_message_text(data)

    print(f"[WEBHOOK] instance={instance_name} tenant_id={tenant_id} chat_id={chat_id} message={message}")

    if chat_id and message:
        try:
            payload = await _process_chat_message(
                tenant_id=tenant_id,
                message=message,
                phone=chat_id,
                session_id=chat_id,
            )
            messages = payload.get('messages') if isinstance(payload.get('messages'), list) else None
            if messages:
                for idx, msg in enumerate(messages):
                    text = str(msg or '').strip()
                    if not text:
                        continue
                    if idx > 0:
                        await asyncio.sleep(1.5)
                    try:
                        await asyncio.to_thread(
                            send_whatsapp_message,
                            chat_id,
                            text,
                            instance_name,
                        )
                    except Exception as exc:
                        logger.error('[SEND] Falha ao enviar mensagem %s no webhook: %s', idx + 1, exc, exc_info=True)
                        continue
            else:
                reply = str(payload.get('response') or payload.get('reply') or '').strip()
                if reply:
                    await asyncio.to_thread(
                        send_whatsapp_message,
                        chat_id,
                        reply,
                        instance_name,
                    )
        except Exception as exc:
            logger.error('[WEBHOOK] erro ao processar mensagem do tenant %s: %s', tenant_id, exc, exc_info=True)

    return {'status': 'ok'}


# ─── Cache invalidation ───────────────────────────────────────────────────────

@app.delete('/api/internal/cache/stock/{tenant_id}')
async def invalidate_stock_cache(tenant_id: str):
    """
    Invalida o cache de estoque de um tenant no Redis.
    Chamado pelas server actions do Next.js após qualquer update de quantidade.
    """
    tenant_id = (tenant_id or '').strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail='tenant_id é obrigatório.')

    r = _get_redis_sync()
    if r:
        try:
            deleted = r.delete(f'stock:{tenant_id}')
            print(f'[CACHE] stock:{tenant_id} invalidado (deleted={deleted})')
        except Exception as e:
            print(f'[CACHE] Falha ao invalidar stock:{tenant_id}: {e}')
            raise HTTPException(status_code=500, detail='Falha ao invalidar cache.')

    return {'ok': True, 'key': f'stock:{tenant_id}'}
