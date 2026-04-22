#!/usr/bin/env python3
"""Discover Supabase schema - find the correct table name for tenants."""
import asyncio
import asyncpg
import json

DB_URI = "postgresql://postgres.dcvtzebchvqktnoojdfw:irMfHHOFlz7ReN27@aws-1-us-west-2.pooler.supabase.com:5432/postgres"

async def discover_schema():
    conn = await asyncpg.connect(DB_URI)
    try:
        # List all tables in public schema
        print("📋 Tables in public schema:")
        rows = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema='public' 
            ORDER BY table_name
        """)
        
        for row in rows:
            table_name = row['table_name']
            print(f"  • {table_name}")
            
            # Count rows in each table to identify likely tenant table
            try:
                count = await conn.fetchval(f'SELECT COUNT(*) FROM "{table_name}"')
                print(f"      ({count} rows)")
            except:
                pass
        
        print("\n\n🔍 Searching for tenant/user/account related tables:")
        search_terms = ['tenant', 'user', 'account', 'client', 'empresa']
        
        for term in search_terms:
            try:
                # Try to find the table
                result = await conn.fetch(f"""
                    SELECT id, * 
                    FROM "{term}" 
                    LIMIT 3
                """)
                print(f"\n✅ Found table: '{term}' ({len(result)} rows shown)")
                if result:
                    print(f"   Columns: {list(result[0].keys())}")
                    # Try to get id, nome, config_nicho
                    for row in result:
                        data = dict(row)
                        print(f"   Row: id={data.get('id')}, nome={data.get('nome', 'N/A')}")
            except Exception as e:
                pass
        
        # Try lowercase variants
        print("\n\n🔄 Trying lowercase variants:")
        for table in ['tenants', 'usuarios', 'users', 'accounts']:
            try:
                result = await conn.fetch(f'SELECT id FROM "{table}" LIMIT 1')
                print(f"✅ Table '{table}' exists!")
                
                # Get column names
                columns = await conn.fetch(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='{table}'
                """)
                col_names = [c['column_name'] for c in columns]
                print(f"   Columns: {col_names}")
                
                # Try to get sample data
                sample = await conn.fetch(f'SELECT * FROM "{table}" LIMIT 2')
                if sample:
                    print(f"   Sample data:")
                    for row in sample:
                        print(f"     {dict(row)}")
                
            except Exception as e:
                print(f"❌ Table '{table}' not found: {str(e)[:60]}")
    
    finally:
        await conn.close()

asyncio.run(discover_schema())
