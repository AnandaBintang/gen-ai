# embedder.py
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss.index"
CHUNKS_PATH = "index/chunks.pkl"
KB_PATH = "data/knowledge_base.json"

class UMKMEmbedder:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks = []

    def load_knowledge_base(self):
        """Load JSON knowledge base"""
        with open(KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def build_index(self):
        """Build FAISS index dari knowledge base"""
        Path("index").mkdir(exist_ok=True)
        data = self.load_knowledge_base()

        # Buat teks representasi untuk setiap dokumen
        self.chunks = []
        texts = []

        for item in data:
            # Teks yang di-embed: gabungan metadata
            embed_text = f"{item['kategori']} {item['nama_produk']} {item['keunggulan']}"
            texts.append(embed_text)
            self.chunks.append(item)

        print(f"Encoding {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Normalisasi untuk cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index (Inner Product = cosine setelah normalisasi)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Simpan index dan chunks
        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Index built: {self.index.ntotal} vectors, dim={dimension}")
        return self.index

    def load_index(self):
        """Load FAISS index yang sudah ada"""
        if not Path(INDEX_PATH).exists():
            print("Index not found, building...")
            return self.build_index()

        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Index loaded: {self.index.ntotal} vectors")
        return self.index

    def retrieve(self, query: str, k: int = 3):
        """Retrieve top-k dokumen paling relevan"""
        if self.index is None:
            self.load_index()

        # Encode query
        query_vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        # Search
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "score": float(score),
                    "document": self.chunks[idx]
                })

        return results


if __name__ == "__main__":
    embedder = UMKMEmbedder()
    embedder.build_index()

    # Test retrieval
    results = embedder.retrieve("sambal pedas homemade makanan", k=3)
    for r in results:
        print(f"Score: {r['score']:.4f} | {r['document']['nama_produk']}")
