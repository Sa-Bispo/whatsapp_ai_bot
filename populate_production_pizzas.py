#!/usr/bin/env python3
"""Populate production pizzaria tenant with proper pizza products."""
import asyncio
import asyncpg
import uuid

DB_URI = "postgresql://postgres.dcvtzebchvqktnoojdfw:irMfHHOFlz7ReN27@aws-1-us-west-2.pooler.supabase.com:5432/postgres"

PRODUCTION_TENANT_ID = "44e290e6-453f-4603-a5b7-ed44d399b03d"

# Pizza flavors with correct pricing
PIZZAS = [
    {"nome": "Calabresa", "categoria": "Tradicional", "preco": 35},
    {"nome": "Frango c/ Catupiry", "categoria": "Tradicional", "preco": 38},
    {"nome": "4 Queijos", "categoria": "Especial", "preco": 42},
    {"nome": "Portuguesa", "categoria": "Tradicional", "preco": 38},
]

async def populate_pizzas():
    conn = await asyncpg.connect(DB_URI)
    try:
        print(f"🍕 Populating production pizzaria tenant: {PRODUCTION_TENANT_ID}\n")
        
        # First, delete existing pizzas to avoid duplicates
        deleted = await conn.execute(
            'DELETE FROM stock_items WHERE tenant_id = $1 AND variacao IN ($2, $3, $4)',
            PRODUCTION_TENANT_ID,
            'Tradicional',
            'Especial',
            'Doce'
        )
        print(f"🗑️  Removed {deleted} old pizza entries\n")
        
        # Insert new pizzas
        for pizza in PIZZAS:
            await conn.execute(
                '''
                INSERT INTO stock_items (id, tenant_id, nome, variacao, preco, ativo)
                VALUES ($1, $2, $3, $4, $5, TRUE)
                ''',
                uuid.uuid4(),
                PRODUCTION_TENANT_ID,
                pizza["nome"],
                pizza["categoria"],
                pizza["preco"],
            )
            print(f"✅ Added: {pizza['nome']} | R${pizza['preco']} | {pizza['categoria']}")
        
        print(f"\n✨ Production pizzaria tenant updated successfully!")
        
        # Verify what we have now
        rows = await conn.fetch('''
            SELECT nome, variacao, preco, ativo 
            FROM stock_items 
            WHERE tenant_id = $1
            ORDER BY nome ASC
        ''', PRODUCTION_TENANT_ID)
        
        print(f"\n📋 Final inventory ({len(rows)} items):")
        for r in rows:
            d = dict(r)
            print(f"   • {d['nome']:25} | R${d['preco']:6.2f} | {d['variacao']:12} | {'✓' if d['ativo'] else '✗'}")
    
    finally:
        await conn.close()

asyncio.run(populate_pizzas())
