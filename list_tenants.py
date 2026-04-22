#!/usr/bin/env python3
"""Listar tenants do Supabase."""
import asyncio
import asyncpg
import json

DB_URI = "postgresql://postgres.dcvtzebchvqktnoojdfw:irMfHHOFlz7ReN27@aws-1-us-west-2.pooler.supabase.com:5432/postgres"

async def list_tenants():
    conn = await asyncpg.connect(DB_URI)
    try:
        rows = await conn.fetch(
            '''
            SELECT id, nome, config_nicho
            FROM "Tenant"
            ORDER BY "createdAt" DESC
            LIMIT 10
            '''
        )
        print(f"Found {len(rows)} tenants:\n")
        for row in rows:
            tenant_id = row['id']
            nome = row['nome']
            config = row['config_nicho']
            print(f"🔹 ID: {tenant_id}")
            print(f"   Nome: {nome}")
            print(f"   Config: {config}")
            print()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await conn.close()

asyncio.run(list_tenants())
