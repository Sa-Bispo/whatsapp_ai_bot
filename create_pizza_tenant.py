#!/usr/bin/env python3
"""Script para criar um tenant de pizzaria de teste no Supabase."""
import asyncio
import json
import asyncpg
import uuid
from datetime import datetime

# Connection string from env - same as BOT_DATABASE_CONNECTION_URI
DB_URI = "postgresql://postgres.dcvtzebchvqktnoojdfw:irMfHHOFlz7ReN27@aws-1-us-west-2.pooler.supabase.com:5432/postgres"


async def create_pizza_tenant():
    """Create a pizza shop test tenant with all related data."""
    
    conn = await asyncpg.connect(DB_URI)
    try:
        async with conn.transaction():
            # Step 0: Get or create a test user
            user_id = '11111111-1111-1111-1111-111111111111'  # Use a test UUID
            existing_user = await conn.fetchval(
                'SELECT id FROM users WHERE id = $1',
                user_id
            )
            
            if not existing_user:
                await conn.execute(
                    '''
                    INSERT INTO users (id, email, nome, data_criacao)
                    VALUES ($1, $2, $3, $4)
                    ''',
                    user_id,
                    'pizzaria_test@test.com',
                    'Pizzaria Teste',
                    datetime.utcnow()
                )
                print(f"✅ User test criado: {user_id}")
            else:
                print(f"✅ User test já existe: {user_id}")
            
            # Step 1: Create tenant
            tenant_id = str(uuid.uuid4())
            config_nicho = {
                "sub_nicho": "pizzaria",
                "nome_negocio": "Pizzaria do João Teste",
                "nome_atendente": "João",
                "faz_delivery": True,
                "faz_retirada": True,
                "horarios": [
                    {
                        "nome": "Jantar",
                        "ativo": True,
                        "dias": [0, 3, 4, 5, 6],
                        "abertura": "18:00",
                        "fechamento": "23:30",
                        "abertura_minutos": 1080,
                        "fechamento_minutos": 1410
                    }
                ]
            }
            
            await conn.execute(
                '''
                INSERT INTO tenants (id, nome, "companyName", "botName", config_nicho, 
                                     user_id, data_criacao)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ''',
                tenant_id,
                'Pizzaria do João Teste',
                'Pizzaria do João Teste',
                'João',
                json.dumps(config_nicho),
                user_id,
                datetime.utcnow()
            )
            
            print(f"✅ Tenant criado: {tenant_id}")
            
            # Step 2: Create pizza flavors (sabores)
            sabores = [
                ('Calabresa', 'Tradicional', 35.00),
                ('Frango c/ Catupiry', 'Tradicional', 38.00),
                ('4 Queijos', 'Especial', 42.00),
                ('Portuguesa', 'Tradicional', 38.00),
            ]
            
            for nome, variacao, preco in sabores:
                await conn.execute(
                    '''
                    INSERT INTO stock_items (id, tenant_id, nome, variacao, preco, 
                                            quantidade, ativo, data_criacao)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ''',
                    str(uuid.uuid4()),
                    tenant_id,
                    nome,
                    variacao,
                    preco,
                    100,  # quantidade
                    True,
                    datetime.utcnow()
                )
            
            print(f"✅ Sabores criados: {[s[0] for s in sabores]}")
            
            # Step 3: Create pizza sizes (tamanhos)
            tamanhos = [
                ('P', 'Pequena', 4, 0, 1),
                ('M', 'Média', 6, 10, 2),
                ('G', 'Grande', 8, 20, 3),
                ('GG', 'GG', 12, 35, 4),
            ]
            
            for sigla, nome, fatias, modificador, ordem in tamanhos:
                await conn.execute(
                    '''
                    INSERT INTO pizza_tamanhos (id, tenant_id, sigla, nome, fatias, 
                                               modificador_preco, ordem, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ''',
                    str(uuid.uuid4()),
                    tenant_id,
                    sigla,
                    nome,
                    fatias,
                    modificador,
                    ordem,
                    datetime.utcnow()
                )
            
            print(f"✅ Tamanhos criados: {[t[0] for t in tamanhos]}")
            
            # Step 4: Create pizza edges (bordas)
            bordas = [
                ('Sem borda', 0),
                ('Borda de catupiry', 8),
                ('Borda de cheddar', 8),
            ]
            
            for nome, preco_extra in bordas:
                await conn.execute(
                    '''
                    INSERT INTO pizza_bordas (id, tenant_id, nome, preco_extra, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ''',
                    str(uuid.uuid4()),
                    tenant_id,
                    nome,
                    preco_extra,
                    datetime.utcnow()
                )
            
            print(f"✅ Bordas criadas: {[b[0] for b in bordas]}")
            
    finally:
        await conn.close()
    
    return tenant_id


if __name__ == '__main__':
    tenant_id = asyncio.run(create_pizza_tenant())
    print(f"\n🎉 Tenant de pizzaria criado com sucesso!")
    print(f"📱 Use este tenant_id para os testes:")
    print(f"   {tenant_id}")
