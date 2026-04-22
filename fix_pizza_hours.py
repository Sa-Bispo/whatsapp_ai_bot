#!/usr/bin/env python3
"""Fix pizza tenant to be open all days."""
import asyncio
import asyncpg
import json
from datetime import datetime

DB_URI = "postgresql://postgres.dcvtzebchvqktnoojdfw:irMfHHOFlz7ReN27@aws-1-us-west-2.pooler.supabase.com:5432/postgres"

async def fix_pizza_tenant():
    conn = await asyncpg.connect(DB_URI)
    try:
        # Update tenant to be open all days
        config_nicho = {
            "sub_nicho": "pizzaria",
            "nome_negocio": "Pizzaria do João Teste",
            "nome_atendente": "João",
            "faz_delivery": True,
            "faz_retirada": True,
            "horarios": [
                {
                    "nome": "Aberto todos os dias",
                    "ativo": True,
                    "dias": [0, 1, 2, 3, 4, 5, 6],  # All days
                    "abertura": "10:00",
                    "fechamento": "23:59",
                    "abertura_minutos": 600,
                    "fechamento_minutos": 1439
                }
            ]
        }
        
        await conn.execute(
            'UPDATE tenants SET config_nicho = $1 WHERE id = $2',
            json.dumps(config_nicho),
            '9b699fc6-ef57-4b04-8f80-b6c6e2d3925e'
        )
        print("✅ Tenant atualizado para estar aberto todos os dias")
    finally:
        await conn.close()

asyncio.run(fix_pizza_tenant())
