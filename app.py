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
from message_buffer import buffer_message
from pizza_flow import process_pizza_message, PizzaState
from config import GEMINI_API_KEY
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


def _build_quick_checkout_summary(order_data: dict, stock_data: dict) -> str:
    items = order_data.get('items') or []
    address = str(order_data.get('customer_address') or '').strip()
    payment = str(order_data.get('payment_method') or '').strip()

    if not items or not address or not payment:
        return ''

    price_map: dict[str, float] = {}
    for item in stock_data.get('items', []) if isinstance(stock_data, dict) else []:
        item_name = _normalize_lookup(str(item.get('nome') or '').strip())
        if not item_name:
            continue
        try:
            price_map[item_name] = float(item.get('preco') or 0)
        except (TypeError, ValueError):
            continue

    lines: list[str] = []
    total = 0.0
    for item in items:
        product_name = str(item.get('base_product_name') or item.get('product_name') or '').strip()
        qty = max(1, int(item.get('quantity') or 1))
        unit_price = 0.0
        normalized_name = _normalize_lookup(re.sub(r'\s*\(.*\)$', '', product_name))
        if normalized_name in price_map:
            unit_price = price_map[normalized_name]
        if unit_price <= 0:
            for candidate_name, candidate_price in price_map.items():
                if candidate_name in normalized_name or normalized_name in candidate_name:
                    unit_price = candidate_price
                    break

        line_total = unit_price * qty
        total += line_total
        if qty > 1:
            lines.append(f'🍺 {qty}x {product_name} — R${unit_price:.2f} cada'.replace('.', ','))
        else:
            lines.append(f'🍺 {product_name}')

    total_line = f"💰 *Total: R${total:.2f}*\n\n".replace('.', ',') if total > 0 else ''
    items_block = '\n'.join(lines) if lines else '🍺 Itens confirmados'

    return (
        '✅ *Pedido anotado!*\n\n'
        f'{items_block}\n'
        f'{total_line}'
        f'📍 {address}\n'
        f'💳 {payment}\n\n'
        'Tô separando já! Em breve saiu 🚀'
    )


def _find_catalog_product_in_message(message: str, stock_data: dict) -> str:
    normalized_message = _normalize_lookup(message)
    if not normalized_message:
        return ''

    best_match = ''
    best_score = 0.0
    message_tokens = set(normalized_message.split())

    for item in stock_data.get('items', []) if isinstance(stock_data, dict) else []:
        name = str(item.get('nome') or '').strip()
        if not name:
            continue
        normalized_name = _normalize_lookup(name)
        if not normalized_name:
            continue

        if normalized_name in normalized_message:
            return name

        name_tokens = [tok for tok in normalized_name.split() if tok and not tok[0].isdigit()]
        if not name_tokens:
            continue

        overlap = len(message_tokens.intersection(set(name_tokens)))
        score = overlap / max(1, len(name_tokens))
        if score > best_score:
            best_score = score
            best_match = name

    return best_match if best_score >= 0.5 else ''


def _extract_payment_from_message(message: str) -> str:
    normalized = (message or '').lower()
    if 'pix' in normalized:
        return 'Pix'
    if any(token in normalized for token in ('cartao', 'cartão', 'credito', 'crédito', 'debito', 'débito')):
        return 'Cartão'
    if 'dinheiro' in normalized:
        return 'Dinheiro'
    return ''


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


def _recover_item_from_history_messages(history_messages: list) -> tuple[str, int]:
    for msg in reversed(history_messages[-10:]):
        if getattr(msg, 'type', '') != 'ai':
            continue

        content = str(getattr(msg, 'content', '') or '').strip()
        if not content:
            continue

        anotado_match = re.search(
            r'(?i)anotado\s+(\d+)\s+(.+?)(?:\s*🍺|\s+me manda o endere[cç]o|\?|\.|$)',
            content,
        )
        if anotado_match:
            qty = max(1, int(anotado_match.group(1)))
            name = anotado_match.group(2).strip(' ,.-')
            if name:
                return name, qty

        confirm_match = re.search(
            r'(?i)temos\s+sim,\s*(.+?),\s*certo\??',
            content,
        )
        if confirm_match:
            name = confirm_match.group(1).strip(' ,.-')
            if name:
                return name, 1

    return '', 0


def _recover_address_from_history_messages(history_messages: list) -> str:
    for msg in reversed(history_messages[-10:]):
        if getattr(msg, 'type', '') != 'human':
            continue
        address = _extract_address_from_message(str(getattr(msg, 'content', '') or ''))
        if address:
            return address
    return ''

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

@app.post('/api/chat-sync')
async def chat_sync(request: Request):
    """Porta síncrona para o simulador da Landing Page.
    Recebe tenant_id + message, carrega a persona do banco e devolve a
    resposta da IA sem disparar nada na Evolution API.
    Inclui roteador de script para intenções pré-IA.
    """
    t0 = time.time()

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
        configs = await get_tenant_configs(tenant_id)
        logger.info('[ROUTER] sub_nicho: %s', configs.get('sub_nicho'))
        conversation_id = f'{tenant_id}:{session_id}'
        
        # PARTE 1: ROTEADOR — verificar intenções pré-IA
        # 1.1 — Verificar horário
        horarios = configs.get('horarios', [])
        if horarios and not is_within_hours(horarios):
            reply = resposta_fora_horario(configs)
            return {'reply': reply, 'response': reply, 'sale_complete': False, 'confetti': False}

        # 1.2 — Detectar intenção com contexto (considera sessão pizzaria)
        import redis
        redis_async = redis.asyncio.from_url(
            "redis://localhost:6379/6", 
            decode_responses=True
        )
        intent = await detect_intent_with_context(
            text=message,
            tenant_id=tenant_id,
            phone=session_id,
            sub_nicho=str(configs.get('sub_nicho') or '').strip().lower(),
            redis_client=redis_async
        )
        # 1.3 — Responder com script se intenção detectada
        if intent:
            if intent == 'saudacao':
                # Só responde saudações na primeira mensagem
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
                # phone = session_id (pode ser phone ou tenant_id)
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
        
        # PARTE 2: FLUXO IA NORMAL — nenhum script bateu
        # PARTE 2.1: Se é pizzaria, usar máquina de estados
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
            
            try:
                estoque = await fetch_stock_for_context(tenant_id)
            except Exception:
                estoque = {}
            
            cardapio = estoque.get('sabores', []) if isinstance(estoque, dict) else []
            tamanhos = estoque.get('tamanhos', []) if isinstance(estoque, dict) else []
            bordas = estoque.get('bordas', []) if isinstance(estoque, dict) else []
            
            logger.info(f"[PIZZA-CHAT-SYNC] tenant_id: {tenant_id}")
            logger.info(f"[PIZZA-CHAT-SYNC] cardapio: {len(cardapio)} sabores")
            logger.info(f"[PIZZA-CHAT-SYNC] tamanhos: {tamanhos}")
            logger.info(f"[PIZZA-CHAT-SYNC] bordas: {bordas}")
            
            reply, session_atualizada = process_pizza_message(
                text=message,
                session=session,
                cardapio=cardapio,
                tamanhos=tamanhos,
                bordas=bordas,
                tenant_config=configs,
            )

            state_value = (session_atualizada or {}).get('state')
            logger.info(f"[CHAT-SYNC] sale_complete check: state={state_value}")
            logger.info(f"[CHAT-SYNC] reply contém ✅: {'✅' in reply}")

            sale_complete = (
                state_value == PizzaState.FINALIZADO.value
                or '✅' in reply
                or 'Pedido confirmado' in reply
            )

            if sale_complete and session_atualizada and not session_atualizada.get('pedido_salvo'):
                try:
                    saved_order = await save_pizza_order(tenant_id=tenant_id, phone=phone, session=session_atualizada)
                    logger.info(f"[CHAT-SYNC] save_pizza_order result: {saved_order}")
                    if saved_order:
                        session_atualizada['pedido_salvo'] = True
                except Exception as exc:
                    logger.error(f"[CHAT-SYNC] erro ao salvar pedido pizzaria: {exc}", exc_info=True)
            
            # Atualizar sessão no Redis
            if redis_client and session_atualizada:
                try:
                    redis_client.setex(session_key, 1800, json.dumps(session_atualizada))
                except Exception:
                    pass

            return {'reply': reply, 'response': reply, 'sale_complete': sale_complete, 'confetti': sale_complete}
        
        # PARTE 2.2: Máquinas de estado para adega e lanchonete
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

            try:
                estoque_data = await fetch_stock_for_context(tenant_id)
            except Exception:
                estoque_data = {}

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
                sale_complete = session_atualizada.get('state') == state_finalizado
                try:
                    if redis_sync:
                        redis_sync.setex(session_key, 1800, json.dumps(session_atualizada, ensure_ascii=False))
                except Exception:
                    pass

                if sale_complete and not session_atualizada.get('pedido_salvo'):
                    try:
                        if sub_nicho == 'adega':
                            p = save_adega_order_payload(session_atualizada)
                            await save_order(
                                tenant_id=tenant_id, phone=phone, nome='Cliente WhatsApp',
                                endereco=p['endereco'],
                                cart_items=[{
                                    'product_name': p['produto'], 'name': p['produto'],
                                    'quantity': p['quantidade'],
                                    'price': p['total'] / max(p['quantidade'], 1),
                                }],
                                total=p['total'], forma_pagamento=p['pagamento'],
                            )
                        else:
                            p = save_lanchonete_order_payload(session_atualizada)
                            await save_order(
                                tenant_id=tenant_id, phone=phone, nome='Cliente WhatsApp',
                                endereco=p['endereco'],
                                cart_items=[
                                    {'product_name': i['nome'], 'name': i['nome'],
                                     'quantity': i['quantidade'], 'price': i['preco']}
                                    for i in p['carrinho']
                                ],
                                total=p['total'], forma_pagamento=p['pagamento'],
                            )
                        session_atualizada['pedido_salvo'] = True
                        if redis_sync:
                            redis_sync.delete(session_key)
                    except Exception as exc:
                        logger.error('[%s-CHAT-SYNC] erro ao salvar pedido: %s', sub_nicho.upper(), exc, exc_info=True)

                return {
                    'reply': reply_flow, 'response': reply_flow,
                    'sale_complete': sale_complete, 'confetti': sale_complete,
                }
            # reply_flow é None → cair no fallback de IA

        prompt_ia = configs.get('promptIa') or ''
        bot_objective = configs.get('botObjective') or 'FECHAR_PEDIDO'

        # generate_persona_response é síncrono (Gemini/OpenAI blocking I/O).
        # Executar em thread pool para não travar o event loop do FastAPI.
        reply: str = await asyncio.to_thread(
            generate_persona_response,
            'Responda à mensagem do usuário de forma natural, seguindo sua persona.',
            message,
            conversation_id,
            prompt_ia or None,
            bot_objective,
            tenant_id,
        )
    except Exception as exc:
        print(f'[chat-sync] erro ao gerar resposta: {exc}')
        return JSONResponse(
            {'error': 'Erro interno ao processar a mensagem.'},
            status_code=500,
        )

    ai_response = (reply or '').rstrip()
    is_sale_complete = ai_response.endswith('✅')

    payload = {
        # Compatibilidade com frontend antigo.
        'reply': ai_response,
        # Novo contrato para o funil de conversão.
        'response': ai_response,
        'sale_complete': is_sale_complete,
        'summary': ai_response if is_sale_complete else None,
        # Sinal opcional para efeitos visuais no frontend.
        'confetti': is_sale_complete,
    }

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
        await buffer_message(
            chat_id=chat_id,
            message=message,
            tenant_id=tenant_id,
            instance_name=instance_name,
        )

    return {'status': 'ok'}


@app.post('/api/produtos/import-vision')
async def import_cardapio_vision(
    tenant_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Endpoint para importar produtos a partir de uma imagem de cardápio.
    
    Recebe:
    - tenant_id: ID do tenant
    - file: Arquivo de imagem (JPG, PNG, etc.)
    
    Retorna:
    - Quantidade de produtos importados
    - Lista de produtos criados (com IDs)
    """
    # Validação básica
    tenant_id = (tenant_id or '').strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail='tenant_id é obrigatório.')

    if not file.filename:
        raise HTTPException(status_code=400, detail='Arquivo não fornecido.')

    # Validar tipo de arquivo (imagem)
    allowed_formats = {'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'}
    if file.content_type not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f'Tipo de arquivo não suportado. Use: JPEG, PNG, GIF, WebP ou BMP.'
        )

    try:
        # Ler conteúdo do arquivo em memória
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail='Arquivo vazio.')

        if gemini_client is None:
            raise HTTPException(
                status_code=500,
                detail='GEMINI_API_KEY não configurada para importação por visão.'
            )

        # ====================================================================
        # Chamada ao Gemini Vision com Fallback de Modelos
        # ====================================================================
        vision_prompt = (
            "Você é um extrator de dados de cardápio. Leia a imagem fornecida. "
            "Retorne EXATAMENTE um array JSON contendo os produtos. "
            "Para cada produto, extraia: 'nome', 'preco_base' (float), 'categoria' (string) "
            "e coloque a descrição do item dentro de 'regras_ia' (string). "
            "Assuma 'ativo': true. "
            "Não retorne markdown, comentários ou texto fora do JSON. "
            "Se não houver produtos legíveis, retorne []."
        )

        # Tentar modelos em ordem de preferência (modelos vision-capable disponíveis)
        models_to_try = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro']
        gemini_response = None
        last_error = None

        for model_name in models_to_try:
            try:
                print(f'[import-vision] Tentando usar modelo: {model_name}')
                gemini_response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        vision_prompt,
                        types.Part.from_bytes(
                            data=file_content,
                            mime_type=file.content_type or 'image/jpeg',
                        ),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type='application/json',
                    ),
                )
                print(f'[import-vision] Modelo {model_name} funcionou com sucesso.')
                break
            except Exception as e:
                last_error = str(e)
                print(f'[import-vision] Erro com modelo {model_name}: {last_error}')
                continue

        if gemini_response is None:
            raise HTTPException(
                status_code=500,
                detail=f'Falha ao processar imagem com Gemini. Último erro: {last_error}'
            )

        # Extrair conteúdo da resposta com segurança
        answer_text = (getattr(gemini_response, 'text', None) or '').strip()
        
        if not answer_text:
            print('[import-vision] Gemini retornou resposta vazia')
            raise HTTPException(
                status_code=500,
                detail='A IA não conseguiu processar a imagem. Verifique a qualidade.'
            )
        
        print(f'[import-vision] Resposta bruta do Gemini (primeiros 500 chars): {answer_text[:500]}')
        
        # Remover markdown code blocks (```json ... ```) se presentes
        if answer_text.startswith('```json'):
            answer_text = answer_text[len('```json'):].strip()
        if answer_text.startswith('```'):
            answer_text = answer_text[len('```'):].strip()
        if answer_text.endswith('```'):
            answer_text = answer_text[:-3].strip()

        # Parsear JSON retornado
        try:
            parsed_data = json.loads(answer_text)
            print(f'[import-vision] JSON parseado com sucesso. Tipo: {type(parsed_data).__name__}')
            
            # Se for lista direta, converter para formato esperado
            if isinstance(parsed_data, list):
                print(f'[import-vision] Resposta é array direto, convertendo para formato esperado.')
                parsed_data = {'produtos': parsed_data}
            
            # Se for string, tentar parsear novamente
            elif isinstance(parsed_data, str):
                print(f'[import-vision] Resposta é string, tentando parsear novamente.')
                parsed_data = json.loads(parsed_data)
                if isinstance(parsed_data, list):
                    parsed_data = {'produtos': parsed_data}
                    
        except json.JSONDecodeError as e:
            print(f'[import-vision] Erro ao fazer parse do JSON da IA: {answer_text}')
            raise HTTPException(
                status_code=500,
                detail=f'Falha ao processar resposta da IA: JSON inválido. {str(e)}'
            )

        # Validar com Pydantic
        try:
            validated = ProdutosVisionList(**parsed_data)
        except Exception as e:
            print(f'[import-vision] Erro ao validar schema Pydantic: {str(e)}')
            raise HTTPException(
                status_code=500,
                detail=f'Resposta da IA não segue o formato esperado: {str(e)}'
            )

        # ====================================================================
        # Salvar produtos no banco de dados
        # ====================================================================
        created_produtos = []
        extracted_produtos = []

        for produto_data in validated.produtos:
            # Sempre retornar a extração para revisão no frontend,
            # mesmo se ocorrer falha ao persistir no banco.
            fallback_id = f'vision_{uuid.uuid4().hex[:8]}'
            extracted_produtos.append({
                'id': fallback_id,
                'nome': produto_data.nome,
                'preco_base': produto_data.preco_base,
                'categoria': produto_data.categoria,
            })

            try:
                # Chamada async para criar produto
                produto_id = await create_produto(
                    tenant_id=tenant_id,
                    nome=produto_data.nome,
                    categoria=produto_data.categoria or 'Geral',
                    preco_base=float(produto_data.preco_base),
                    classe_negocio='generico',
                    regras_ia=produto_data.regras_ia or '',
                    config_nicho={},
                )
                
                if produto_id:
                    created_produtos.append({
                        'id': produto_id,
                        'nome': produto_data.nome,
                        'preco_base': produto_data.preco_base,
                        'categoria': produto_data.categoria,
                    })
            except Exception as e:
                print(f'[import-vision] Erro ao criar produto "{produto_data.nome}": {str(e)}')
                # Continuar com próximos produtos em caso de erro individual
                continue

        # Retornar resultado
        produtos_resposta = created_produtos if len(created_produtos) > 0 else extracted_produtos
        return JSONResponse({
            'sucesso': True,
            'quantidade_importada': len(produtos_resposta),
            'produtos': produtos_resposta,
            'mensagem': f'{len(produtos_resposta)} produtos extraidos com sucesso para revisao.',
        })

    except HTTPException:
        raise
    except Exception as exc:
        print(f'[import-vision] Erro geral: {str(exc)}')
        raise HTTPException(
            status_code=500,
            detail=f'Erro ao processar a imagem: {str(exc)}'
        )


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
