import chromadb
from config import CHROMA_PERSIST_DIR


class VectorDBManager:

    def __init__(
        self, db_path: str = CHROMA_PERSIST_DIR, collection_name: str = "movie_shots"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client: chromadb.PersistentClient | None = None
        self.collection = None

    def connect(self):
        """Conecta a la base de datos ChromaDB existente."""
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  
        )

    def disconnect(self):
        """Libera los recursos de la conexión."""
        self.client = None
        self.collection = None

    def search_similar(
        self,
        query_vector: list,
        n_results: int = 3,
        movie_title: str | None = None,
    ):
        """Busca los frames más similares al vector de texto de la consulta."""
        if not self.collection:
            raise RuntimeError(
                "La base de datos no está conectada. Llama a connect() primero."
            )

        total_count = self.collection.count()
        if total_count == 0:
            return []

        where_clause = (
            {"movie_title": movie_title} if movie_title else None
        )

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(n_results, total_count),
            where=where_clause,
        )

        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = (
                results["metadatas"][0]
                if results.get("metadatas")
                else [{}] * len(ids)
            )
            distances = (
                results["distances"][0]
                if results.get("distances")
                else [0.0] * len(ids)
            )

            for shot_id, metadata, distance in zip(ids, metadatas, distances):
                # Conversión de distancia coseno a score de similitud (0.0 a 1.0)
                similarity_score = max(0.0, 1.0 - (float(distance) / 2.0))

                formatted_results.append({
                    "id": shot_id,
                    "score": round(similarity_score, 4),
                    "metadata": metadata,
                })

        return formatted_results


# Instancia única para usar en la aplicación
db_manager = VectorDBManager()