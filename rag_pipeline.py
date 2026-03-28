"""
RAG Pipeline: embeddings, vector store (FAISS), retrieval en generatie.
"""

import pickle
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Module-level cache so the same model is not loaded twice
_sentence_transformer_cache: dict = {}


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if model_name not in _sentence_transformer_cache:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _sentence_transformer_cache[model_name] = SentenceTransformer(model_name)
        self.model = _sentence_transformer_cache[model_name]
        self.dimensie = self.model.get_sentence_embedding_dimension()

    def embed(self, teksten: List[str], batch_grootte: int = 32):
        """Embed a list of texts, returning a float32 numpy array."""
        import numpy as np
        embeddings = self.model.encode(
            teksten,
            batch_size=batch_grootte,
            show_progress_bar=len(teksten) > 50,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class VectorStore:
    def __init__(self, dimensie: int):
        import faiss
        self.dimensie = dimensie
        self.index = faiss.IndexFlatIP(dimensie)
        self.chunks: List[Dict] = []

    @property
    def grootte(self) -> int:
        return self.index.ntotal

    def voeg_toe(self, embeddings, chunks: List[Dict]) -> None:
        assert embeddings.shape[0] == len(chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def zoek(self, query_embedding, k: int = 5) -> List[Tuple[Dict, float]]:
        if self.grootte == 0:
            return []
        k = min(k, self.grootte)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def opslaan(self, map: str = "data") -> None:
        import faiss
        Path(map).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{map}/faiss_index.bin")
        with open(f"{map}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def laden(self, map: str = "data") -> bool:
        import faiss
        index_pad = f"{map}/faiss_index.bin"
        chunks_pad = f"{map}/chunks.pkl"
        if not (Path(index_pad).exists() and Path(chunks_pad).exists()):
            return False
        self.index = faiss.read_index(index_pad)
        with open(chunks_pad, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Vector store loaded: {self.grootte} chunks")
        return True

    def wis(self) -> None:
        import faiss
        self.index = faiss.IndexFlatIP(self.dimensie)
        self.chunks = []


class LLMClient:
    def __init__(self, token: str, model: str = LLM_MODEL):
        from huggingface_hub import InferenceClient
        self.model = model
        self.client = InferenceClient(token=token)

    def genereer(
        self,
        vraag: str,
        context_chunks: List[Tuple[Dict, float]],
        max_tokens: int = 512,
        temperatuur: float = 0.1,
    ) -> str:
        """Generate an answer given a question and retrieved context chunks."""
        context = self._bouw_context(context_chunks)
        prompt = self._bouw_prompt(vraag, context)
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperatuur,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Fout bij genereren antwoord: {e}"

    def _bouw_context(self, chunks: List[Tuple[Dict, float]]) -> str:
        delen = []
        for chunk, _ in chunks:
            delen.append(
                f"[{chunk.get('titel', '')} - {chunk.get('artikel', '')}]\n{chunk.get('tekst', '')}"
            )
        return "\n\n---\n\n".join(delen)

    def _bouw_prompt(self, vraag: str, context: str) -> str:
        return (
            "Je bent een behulpzame juridische assistent gespecialiseerd in Nederlandse wetgeving.\n"
            "Beantwoord de vraag uitsluitend op basis van de onderstaande wetteksten.\n"
            "Als het antwoord niet in de context staat, zeg dit dan eerlijk.\n"
            "Antwoord altijd in het Nederlands en verwijs naar het relevante wetsartikel.\n\n"
            f"WETTEKSTEN:\n{context}\n\n"
            f"VRAAG: {vraag}\n\n"
            "ANTWOORD:"
        )


class RAGPipeline:
    def __init__(
        self,
        hf_token: str = "",
        data_map: str = "data",
        llm_model: str = LLM_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.data_map = data_map
        self._hf_token = hf_token
        self._llm_model = llm_model
        self.embedding_model = EmbeddingModel(embedding_model)
        self.vector_store = VectorStore(dimensie=self.embedding_model.dimensie)
        self.llm: Optional[LLMClient] = None
        if hf_token:
            self.stel_llm_in(hf_token, llm_model)

    def stel_llm_in(self, hf_token: str, model: str = LLM_MODEL) -> None:
        """Initialise or replace the LLM client."""
        self._hf_token = hf_token
        self._llm_model = model
        self.llm = LLMClient(token=hf_token, model=model)

    def voeg_chunks_toe(self, chunks: List[Dict]) -> int:
        """Embed and add chunks to the FAISS index."""
        if not chunks:
            return 0
        teksten = [c["tekst"] for c in chunks]
        embeddings = self.embedding_model.embed(teksten)
        self.vector_store.voeg_toe(embeddings, chunks)
        self.vector_store.opslaan(self.data_map)
        return len(chunks)

    def laden(self) -> bool:
        """Load a previously saved index from disk."""
        return self.vector_store.laden(self.data_map)

    def wis_index(self) -> None:
        """Clear the in-memory vector store and delete persisted files."""
        self.vector_store.wis()
        for bestand in [f"{self.data_map}/faiss_index.bin", f"{self.data_map}/chunks.pkl"]:
            if Path(bestand).exists():
                Path(bestand).unlink()

    def rebuild_index(self, new_embedding_model: str) -> None:
        """Re-embed all stored chunks with a different embedding model.

        This replaces the current EmbeddingModel and VectorStore in-place.
        No-op if the index is empty or the model has not changed.
        """
        if self.vector_store.grootte == 0:
            logger.info("rebuild_index called on empty store; switching model only")
            self.embedding_model = EmbeddingModel(new_embedding_model)
            self.vector_store = VectorStore(dimensie=self.embedding_model.dimensie)
            return

        if new_embedding_model == self.embedding_model.model.tokenizer.name_or_path:
            return  # already using this model

        existing_chunks = list(self.vector_store.chunks)
        self.embedding_model = EmbeddingModel(new_embedding_model)
        self.vector_store = VectorStore(dimensie=self.embedding_model.dimensie)

        teksten = [c["tekst"] for c in existing_chunks]
        embeddings = self.embedding_model.embed(teksten)
        self.vector_store.voeg_toe(embeddings, existing_chunks)
        self.vector_store.opslaan(self.data_map)
        logger.info(f"Index rebuilt with model {new_embedding_model}: {self.vector_store.grootte} chunks")

    def zoek(self, vraag: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve the k most relevant chunks for a question."""
        query_embedding = self.embedding_model.embed([vraag])[0]
        return self.vector_store.zoek(query_embedding, k=k)

    def stel_vraag(self, vraag: str, k: int = 5, max_tokens: int = 512) -> Dict:
        """Full RAG query: retrieve context and generate an answer."""
        if self.vector_store.grootte == 0:
            return {"antwoord": "Geen wetgeving geladen.", "bronnen": [], "context": []}

        relevante_chunks = self.zoek(vraag, k=k)
        if not relevante_chunks:
            return {"antwoord": "Geen relevante wetsartikelen gevonden.", "bronnen": [], "context": []}

        if self.llm:
            antwoord = self.llm.genereer(vraag, relevante_chunks, max_tokens=max_tokens)
        else:
            antwoord = "Geen HuggingFace token ingesteld. Meest relevante artikelen:\n\n"
            for chunk, _ in relevante_chunks[:3]:
                antwoord += f"**{chunk.get('titel', '')} - {chunk.get('artikel', '')}**\n"
                antwoord += chunk.get("tekst", "")[:500] + "...\n\n"

        bronnen = []
        gezien = set()
        for chunk, score in relevante_chunks:
            sleutel = f"{chunk.get('bwbr_id', '')}_{chunk.get('artikel', '')}"
            if sleutel not in gezien:
                gezien.add(sleutel)
                bronnen.append({
                    "titel": chunk.get("titel", ""),
                    "artikel": chunk.get("artikel", ""),
                    "url": chunk.get("url", ""),
                    "score": round(score, 3),
                    "tekst": chunk.get("tekst", "")[:300],
                })

        return {"antwoord": antwoord, "bronnen": bronnen, "context": relevante_chunks}

    def query_with_details(self, vraag: str, k: int = 5, max_tokens: int = 512) -> Dict:
        """Like stel_vraag but also returns retrieved chunks and latency.

        Returns a dict with keys:
        - antwoord (str)
        - bronnen (list)
        - retrieved_chunks (List[Tuple[Dict, float]])
        - latency_ms (float)
        """
        t0 = time.monotonic()
        result = self.stel_vraag(vraag, k=k, max_tokens=max_tokens)
        latency_ms = (time.monotonic() - t0) * 1000
        result["retrieved_chunks"] = result.pop("context", [])
        result["latency_ms"] = round(latency_ms, 1)
        return result

    @property
    def aantal_chunks(self) -> int:
        return self.vector_store.grootte
