"""
RAG Pipeline: embeddings, vector store (FAISS), retrieval en generatie.
Gebruikt sentence-transformers voor embeddings en HuggingFace Inference API voor de LLM.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# Multilinguaal model, goed voor Nederlands, ~50MB
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Vast model — goed voor Nederlands, betrouwbaar op HF gratis tier
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

INDEX_BESTAND = "data/faiss_index.bin"
CHUNKS_BESTAND = "data/chunks.pkl"


@st.cache_resource
def _laad_sentence_transformer(model_naam: str) -> SentenceTransformer:
    """Laad het embedding model eenmalig en cache het over alle sessies."""
    logger.info(f"Laden embedding model: {model_naam}")
    return SentenceTransformer(model_naam)


class EmbeddingModel:
    """Wrapper voor sentence-transformers embedding model."""

    def __init__(self, model_naam: str = EMBEDDING_MODEL):
        self.model = _laad_sentence_transformer(model_naam)
        self.dimensie = self.model.get_sentence_embedding_dimension()

    def embed(self, teksten: List[str], batch_grootte: int = 32) -> np.ndarray:
        """Embed een lijst teksten. Geeft numpy array terug."""
        embeddings = self.model.encode(
            teksten,
            batch_size=batch_grootte,
            show_progress_bar=len(teksten) > 50,
            normalize_embeddings=True,  # Nodig voor cosine similarity via dot product
        )
        return embeddings.astype(np.float32)


class VectorStore:
    """FAISS vector store voor opslaan en zoeken van embeddings."""

    def __init__(self, dimensie: int):
        self.dimensie = dimensie
        # IndexFlatIP met genormaliseerde embeddings = cosine similarity
        self.index = faiss.IndexFlatIP(dimensie)
        self.chunks: List[Dict] = []

    @property
    def grootte(self) -> int:
        return self.index.ntotal

    def voeg_toe(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """Voeg embeddings en bijbehorende chunks toe aan de index."""
        assert embeddings.shape[0] == len(chunks), "Aantal embeddings moet gelijk zijn aan aantal chunks"
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def zoek(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Zoek de k meest relevante chunks.
        Geeft lijst van (chunk, score) tuples terug.
        """
        if self.grootte == 0:
            return []

        k = min(k, self.grootte)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)

        resultaten = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                resultaten.append((self.chunks[idx], float(score)))
        return resultaten

    def opslaan(self, map: str = "data") -> None:
        """Sla index en chunks op naar schijf."""
        Path(map).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{map}/faiss_index.bin")
        with open(f"{map}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Vector store opgeslagen in {map}/")

    def laden(self, map: str = "data") -> bool:
        """Laad index en chunks van schijf. Geeft True terug bij succes."""
        index_pad = f"{map}/faiss_index.bin"
        chunks_pad = f"{map}/chunks.pkl"

        if not (Path(index_pad).exists() and Path(chunks_pad).exists()):
            return False

        self.index = faiss.read_index(index_pad)
        with open(chunks_pad, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Vector store geladen: {self.grootte} chunks")
        return True

    def wis(self) -> None:
        """Verwijder alle chunks en reset de index."""
        self.index = faiss.IndexFlatIP(self.dimensie)
        self.chunks = []


class LLMClient:
    """Client voor HuggingFace Inference API."""

    def __init__(self, token: str):
        # provider="hf-inference" voorkomt doorrouting naar externe providers (bijv. novita)
        self.client = InferenceClient(model=LLM_MODEL, token=token, provider="hf-inference")

    def genereer(
        self,
        vraag: str,
        context_chunks: List[Tuple[Dict, float]],
        max_tokens: int = 512,
        temperatuur: float = 0.1,
    ) -> str:
        context = self._bouw_context(context_chunks)
        prompt = self._bouw_prompt(vraag, context)

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperatuur,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generatie mislukt: {e}")
            return f"Fout bij genereren antwoord: {e}"

    def _bouw_context(self, chunks: List[Tuple[Dict, float]]) -> str:
        """Bouw een gestructureerde context op uit de gevonden chunks."""
        context_delen = []
        for chunk, score in chunks:
            titel = chunk.get("titel", "Onbekende wet")
            artikel = chunk.get("artikel", "")
            tekst = chunk.get("tekst", "")
            context_delen.append(f"[{titel} - {artikel}]\n{tekst}")
        return "\n\n---\n\n".join(context_delen)

    def _bouw_prompt(self, vraag: str, context: str) -> str:
        """Bouw de volledige prompt op in het Nederlands."""
        return f"""Je bent een behulpzame juridische assistent gespecialiseerd in Nederlandse wetgeving.
Beantwoord de vraag uitsluitend op basis van de onderstaande wetteksten.
Als het antwoord niet in de context staat, zeg dit dan eerlijk.
Antwoord altijd in het Nederlands en verwijs naar het relevante wetsartikel.

WETTEKSTEN:
{context}

VRAAG: {vraag}

ANTWOORD:"""


class RAGPipeline:
    """
    Volledige RAG pipeline:
    1. Embeddings via sentence-transformers
    2. Opslag in FAISS vector store
    3. Retrieval van relevante chunks
    4. Generatie via HuggingFace LLM
    """

    def __init__(
        self,
        hf_token: str = "",
        data_map: str = "data",
    ):
        self.data_map = data_map
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(dimensie=self.embedding_model.dimensie)
        self.llm: Optional[LLMClient] = None

        if hf_token:
            self.stel_llm_in(hf_token)

    def stel_llm_in(self, hf_token: str) -> None:
        """Initialiseer de LLM client."""
        self.llm = LLMClient(token=hf_token)
        logger.info(f"LLM ingesteld: {LLM_MODEL}")

    def index_opbouwen(
        self,
        chunks: List[Dict],
        voortgang_callback=None,
    ) -> int:
        """
        Embed alle chunks en sla op in FAISS.
        chunks: lijst van dicts met minimaal 'tekst'.
        Geeft aantal geïndexeerde chunks terug.
        """
        if not chunks:
            return 0

        teksten = [c["tekst"] for c in chunks]

        if voortgang_callback:
            voortgang_callback("Embeddings berekenen...", 0)

        embeddings = self.embedding_model.embed(teksten)

        if voortgang_callback:
            voortgang_callback("Opslaan in vector store...", 0.8)

        self.vector_store.voeg_toe(embeddings, chunks)
        self.vector_store.opslaan(self.data_map)

        if voortgang_callback:
            voortgang_callback("Klaar!", 1.0)

        return len(chunks)

    def voeg_chunks_toe(self, chunks: List[Dict]) -> int:
        """Voeg extra chunks toe aan een bestaande index."""
        if not chunks:
            return 0
        teksten = [c["tekst"] for c in chunks]
        embeddings = self.embedding_model.embed(teksten)
        self.vector_store.voeg_toe(embeddings, chunks)
        self.vector_store.opslaan(self.data_map)
        return len(chunks)

    def laden(self) -> bool:
        """Laad een eerder opgeslagen index."""
        return self.vector_store.laden(self.data_map)

    def wis_index(self) -> None:
        """Verwijder de hele vector store."""
        self.vector_store.wis()
        # Verwijder ook bestanden
        for bestand in [f"{self.data_map}/faiss_index.bin", f"{self.data_map}/chunks.pkl"]:
            if Path(bestand).exists():
                Path(bestand).unlink()

    def zoek(self, vraag: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Zoek de k meest relevante wetsartikelen voor de gegeven vraag.
        Geeft lijst van (chunk, score) tuples terug.
        """
        query_embedding = self.embedding_model.embed([vraag])[0]
        return self.vector_store.zoek(query_embedding, k=k)

    def stel_vraag(
        self,
        vraag: str,
        k: int = 5,
        max_tokens: int = 512,
    ) -> Dict:
        """
        Volledige RAG query: zoek relevante context en genereer antwoord.
        Geeft dict terug met 'antwoord', 'bronnen', 'context'.
        """
        if self.vector_store.grootte == 0:
            return {
                "antwoord": "Geen wetgeving geladen. Laad eerst wetten via het tabblad 'Wetgeving laden'.",
                "bronnen": [],
                "context": [],
            }

        # Stap 1: Zoek relevante chunks
        relevante_chunks = self.zoek(vraag, k=k)

        if not relevante_chunks:
            return {
                "antwoord": "Geen relevante wetsartikelen gevonden voor uw vraag.",
                "bronnen": [],
                "context": [],
            }

        # Stap 2: Genereer antwoord
        if self.llm:
            antwoord = self.llm.genereer(vraag, relevante_chunks, max_tokens=max_tokens)
        else:
            # Geen LLM: toon alleen de gevonden chunks
            antwoord = "⚠️ Geen HuggingFace token ingesteld. Hieronder staan de meest relevante wetsartikelen:\n\n"
            for chunk, score in relevante_chunks[:3]:
                antwoord += f"**{chunk.get('titel', '')} - {chunk.get('artikel', '')}**\n"
                antwoord += chunk.get("tekst", "")[:500] + "...\n\n"

        # Stap 3: Bereid bronnen voor
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

        return {
            "antwoord": antwoord,
            "bronnen": bronnen,
            "context": relevante_chunks,
        }

    @property
    def aantal_chunks(self) -> int:
        return self.vector_store.grootte
