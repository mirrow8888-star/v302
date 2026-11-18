"""建置階段預先下載嵌入模型到專案目錄，避免執行時才下載。EMBED_MODEL=none 時略過。"""
from rag_engine import CACHE_DIR, EMBED_MODEL, NO_EMBED

if NO_EMBED:
    print("EMBED_MODEL=none，略過預載")
else:
    from fastembed import TextEmbedding
    TextEmbedding(model_name=EMBED_MODEL, cache_dir=CACHE_DIR)
    print("embedding model cached:", CACHE_DIR)
