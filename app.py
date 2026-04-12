import asyncio
import json
import re
import uuid

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from chains import generate_persona_response
from database_api import get_tenant_by_instance, get_tenant_configs, create_produto
from message_buffer import buffer_message
from config import GEMINI_API_KEY


app = FastAPI()

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
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({'error': 'JSON inválido.'}, status_code=400)

    tenant_id = (data.get('tenant_id') or '').strip()
    message = (data.get('message') or '').strip()
    # session_id permite manter histórico separado por visitante.
    session_id = (data.get('session_id') or tenant_id).strip()

    if not tenant_id or not message:
        return JSONResponse(
            {'error': 'Os campos tenant_id e message são obrigatórios.'},
            status_code=422,
        )

    try:
        configs = await get_tenant_configs(tenant_id)
        prompt_ia = configs.get('promptIa') or ''
        bot_objective = configs.get('botObjective') or 'FECHAR_PEDIDO'

        # generate_persona_response é síncrono (Gemini/OpenAI blocking I/O).
        # Executar em thread pool para não travar o event loop do FastAPI.
        reply: str = await asyncio.to_thread(
            generate_persona_response,
            'Responda à mensagem do usuário de forma natural, seguindo sua persona.',
            message,
            session_id,
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
