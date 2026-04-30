import asyncio
import os

import asyncpg

TENANT = "aafce803-e21a-41a2-b5f4-7ed614c2a897"
DB = os.getenv("BOT_DATABASE_CONNECTION_URI")


async def main() -> None:
    if not DB:
        raise RuntimeError("BOT_DATABASE_CONNECTION_URI nao configurada")

    conn = await asyncpg.connect(DB)
    try:
        row = await conn.fetchrow(
            "SELECT id FROM stock_items WHERE tenant_id=$1 AND lower(nome)=lower($2) LIMIT 1",
            TENANT,
            "Porção de Batata Frita",
        )
        if row:
            item_id = row["id"]
            await conn.execute(
                "UPDATE stock_items SET variacao=$1, preco=$2, quantidade=$3, ativo=true, tem_variacoes=true WHERE id=$4",
                "Porções",
                0,
                100,
                item_id,
            )
        else:
            item_id = await conn.fetchval(
                "INSERT INTO stock_items (id,nome,variacao,preco,quantidade,ativo,tem_variacoes,tenant_id,data_criacao) VALUES (gen_random_uuid(),$1,$2,$3,$4,true,true,$5,now()) RETURNING id",
                "Porção de Batata Frita",
                "Porções",
                0,
                100,
                TENANT,
            )

        await conn.execute("DELETE FROM item_variacoes WHERE item_id=$1", item_id)
        await conn.execute("DELETE FROM item_adicionais WHERE item_id=$1", item_id)

        for ordem, sigla, nome, preco in [
            (0, "P", "Pequena", 18.0),
            (1, "M", "Média", 24.0),
            (2, "G", "Grande", 32.0),
        ]:
            await conn.execute(
                "INSERT INTO item_variacoes (id,item_id,sigla,nome,preco,ordem,created_at) VALUES (gen_random_uuid(),$1,$2,$3,$4,$5,now())",
                item_id,
                sigla,
                nome,
                preco,
                ordem,
            )

        for nome, preco_extra in [("Bacon", 5.0), ("Cheddar", 4.0), ("Catupiry", 4.0)]:
            await conn.execute(
                "INSERT INTO item_adicionais (id,item_id,nome,preco_extra,ativo,created_at) VALUES (gen_random_uuid(),$1,$2,$3,true,now())",
                item_id,
                nome,
                preco_extra,
            )

        print("Item de teste atualizado:", item_id)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
