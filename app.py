"""
Streamlit RAG-applicatie voor Nederlandse wetgeving.
"""

import streamlit as st

from scraper import WettenScraper, BEKENDE_WETTEN, plat_maken
from rag_pipeline import RAGPipeline

_HF_TOKEN = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""

st.set_page_config(
    page_title="Wetgeving Assistent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "geladen_wetten" not in st.session_state:
    st.session_state.geladen_wetten = []


def laad_pipeline() -> RAGPipeline:
    if st.session_state.pipeline is None:
        with st.spinner("Model laden..."):
            pipeline = RAGPipeline(hf_token=_HF_TOKEN, data_map="data")
            pipeline.laden()
            st.session_state.pipeline = pipeline
    return st.session_state.pipeline


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Wetgeving Assistent")
    st.markdown("---")

    pagina = st.radio(
        "Navigatie",
        ["Assistent", "Architectuur"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if st.session_state.pipeline and st.session_state.pipeline.aantal_chunks > 0:
        st.markdown(f"**{st.session_state.pipeline.aantal_chunks}** chunks geïndexeerd")
        if st.session_state.geladen_wetten:
            st.markdown("**Geladen wetten**")
            for wet in st.session_state.geladen_wetten:
                st.markdown(f"- {wet}")
        st.markdown("---")
        if st.button("Index wissen", type="secondary", use_container_width=True):
            st.session_state.pipeline.wis_index()
            st.session_state.geladen_wetten = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("Geen wetten geladen.")


# ─── Pagina: Assistent ─────────────────────────────────────────────────────
if pagina == "Assistent":
    st.markdown("## Nederlandse Wetgeving Assistent")
    st.markdown(
        "Laad een of meer wetten en stel vervolgens vragen. "
        "De assistent zoekt de relevante artikelen op en genereert een antwoord."
    )
    st.markdown("---")

    # Wetgeving laden
    st.markdown("### Wetgeving laden")
    col1, col2 = st.columns([3, 1])
    with col1:
        geselecteerde_wetten = st.multiselect(
            "Selecteer wetten",
            options=list(BEKENDE_WETTEN.keys()),
            placeholder="Kies een of meer wetten...",
            label_visibility="collapsed",
        )
    with col2:
        handmatig_bwbr = st.text_input(
            "BWB ID",
            placeholder="bijv. BWBR0001840",
            label_visibility="collapsed",
        )

    if st.button("Laden en indexeren", type="primary"):
        te_laden = [BEKENDE_WETTEN[w] for w in geselecteerde_wetten]
        if handmatig_bwbr.strip():
            te_laden.append(handmatig_bwbr.strip())

        if not te_laden:
            st.warning("Selecteer een wet of voer een BWB ID in.")
        else:
            pipeline = laad_pipeline()
            scraper = WettenScraper(delay=0.5)
            alle_chunks = []
            voortgang = st.progress(0)

            for i, bwbr_id in enumerate(te_laden):
                voortgang.progress(i / len(te_laden))
                wet = scraper.haal_wet_op_bwbr(bwbr_id)
                if wet:
                    alle_chunks.extend(plat_maken([wet]))
                    if wet["titel"] not in st.session_state.geladen_wetten:
                        st.session_state.geladen_wetten.append(wet["titel"])
                    st.success(f"{wet['titel']} — {len(wet['artikelen'])} artikelen opgehaald")
                else:
                    st.error(f"Kon {bwbr_id} niet ophalen")

            if alle_chunks:
                voortgang.progress(0.9)
                pipeline.voeg_chunks_toe(alle_chunks)
                voortgang.progress(1.0)
                st.rerun()

    st.markdown("---")

    # Chat
    st.markdown("### Vragen stellen")

    pipeline = laad_pipeline()

    if pipeline.aantal_chunks == 0:
        st.info("Laad eerst een of meer wetten hierboven.")
    else:
        vraag_input = st.chat_input("Stel uw vraag over de Nederlandse wetgeving...")

        if vraag_input:
            st.session_state.chat_history.append({"rol": "gebruiker", "inhoud": vraag_input})
            with st.spinner("Antwoord genereren..."):
                resultaat = pipeline.stel_vraag(vraag_input)
            st.session_state.chat_history.append({
                "rol": "assistent",
                "inhoud": resultaat["antwoord"],
                "bronnen": resultaat.get("bronnen", []),
            })

        for bericht in st.session_state.chat_history:
            if bericht["rol"] == "gebruiker":
                with st.chat_message("user"):
                    st.markdown(bericht["inhoud"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(bericht["inhoud"])
                    if bericht.get("bronnen"):
                        with st.expander(f"Bronnen ({len(bericht['bronnen'])} artikelen)"):
                            for bron in bericht["bronnen"]:
                                st.markdown(
                                    f"**{bron['titel']} – {bron['artikel']}** "
                                    f"*(relevantie: {int(bron['score'] * 100)}%)*"
                                )
                                if bron.get("url"):
                                    st.markdown(f"[Bekijk op wetten.overheid.nl]({bron['url']})")
                                st.markdown(f"> {bron['tekst']}...")
                                st.markdown("---")

        if st.session_state.chat_history:
            if st.button("Gesprek wissen", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()


# ─── Pagina: Architectuur ──────────────────────────────────────────────────
elif pagina == "Architectuur":
    st.markdown("## Architectuur")
    st.markdown(
        "Deze applicatie gebruikt een **RAG (Retrieval-Augmented Generation)** pipeline "
        "om vragen over Nederlandse wetgeving te beantwoorden."
    )
    st.markdown("---")

    st.markdown("### Werking")
    st.markdown("""
1. **Scraper** — haalt wetteksten op van [wetten.overheid.nl](https://wetten.overheid.nl)
   via officiële BWB IDs. De tekst wordt per artikel opgesplitst in chunks.

2. **Embeddings** — elk chunk wordt omgezet naar een vector met het model
   `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, ~50 MB, meertalig, lokaal).

3. **Vector store** — de vectors worden opgeslagen in een FAISS index
   (`IndexFlatIP` met genormaliseerde embeddings = cosine similarity).

4. **Retrieval** — bij een vraag wordt de vraag geëmbed en worden de meest relevante
   artikelen opgezocht via nearest-neighbor search.

5. **Generatie** — de gevonden artikelen worden als context meegegeven aan
   `zephyr-7b-beta` via de HuggingFace Inference API. Het model genereert een
   Nederlands antwoord met verwijzing naar de relevante artikelen.
    """)

    st.markdown("### Componentenoverzicht")
    st.markdown("""
| Component | Technologie |
|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| Vector store | FAISS `IndexFlatIP` |
| Similarity | Cosine (via genormaliseerde embeddings) |
| LLM | `zephyr-7b-beta` via HuggingFace Inference API |
| Chunking | Per wetsartikel, fallback per 1000 tekens |
| Frontend | Streamlit |
    """)

    st.markdown("### Diagram")
    st.code("""
┌─────────────────────────────────────┐
│         Streamlit UI (app.py)       │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────────────────────┐
    │                             │
┌───▼────────────┐    ┌───────────▼──────────────┐
│  scraper.py    │    │  rag_pipeline.py          │
│  HTML → chunks │    │  EmbeddingModel (lokaal)  │
└────────────────┘    │  VectorStore (FAISS)      │
                      │  LLMClient (HF API)       │
                      └──────────────────────────┘
    """, language=None)

    st.markdown("### Beperkingen")
    st.markdown("""
- De FAISS index wordt niet persistent opgeslagen op Streamlit Community Cloud.
  Gebruikers moeten wetten laden bij elke nieuwe sessie.
- De gratis HuggingFace Inference API heeft rate limits bij intensief gebruik.
- Wetteksten worden niet automatisch bijgewerkt. Wis de index en herlaad
  voor de meest actuele versie.
    """)
