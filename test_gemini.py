import asyncio

from dotenv import load_dotenv

from gemini_parser import extract_order_intent


load_dotenv()


catalogo_mock = [
    {
        'codigo_pai': 'WMAX-900',
        'nome_produto': 'Whey Max',
        'variacoes': ['Chocolate', 'Baunilha'],
    },
    {
        'codigo_pai': 'CREA-GROWTH',
        'nome_produto': 'Creatina Growth',
        'variacoes': ['Unico'],
    },
]


async def main() -> None:
    resultado = await extract_order_intent(
        'Manda 2 wheys de chocolate da max e uma creatina',
        catalogo_mock,
    )
    print('Resultado final:', resultado)


if __name__ == '__main__':
    asyncio.run(main())
