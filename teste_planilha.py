from sheets_api import list_estoque

try:
    produtos = list_estoque()
    print("Conexão feita com sucesso! Produtos encontrados:")
    print(produtos)
except Exception as e:
    print(f"Erro ao conectar: {e}")