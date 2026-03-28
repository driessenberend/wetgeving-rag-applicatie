"""
Streamlit RAG-applicatie voor Nederlandse wetgeving.
Stel vragen over Nederlandse wetten via een gratis AI-assistent.
"""

import streamlit as st

from scraper import WettenScraper, BEKENDE_WETTEN, plat_maken
from rag_pipeline import RAGPipeline

# HF token via Streamlit secrets (voor Streamlit Cloud deployment)
_DEFAULT_TOKEN = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""

st.set_page_config(
    page_title="Nederlandse Wetgeving RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Sessie-state initialisatie
# ─────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "geladen_wetten" not in st.session_state:
    st.session_state.geladen_wetten = []


def laad_pipeline(hf_token: str) -> RAGPipeline:
    if st.session_state.pipeline is None:
        with st.spinner("Embedding model laden (eenmalig ~30 seconden)..."):
            pipeline = RAGPipeline(hf_token=hf_token, data_map="data")
            if pipeline.laden():
                st.success(f"Bestaande index geladen ({pipeline.aantal_chunks} chunks)")
            st.session_state.pipeline = pipeline
    else:
        if hf_token:
            st.session_state.pipeline.stel_llm_in(hf_token)
    return st.session_state.pipeline


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Wetgeving RAG")
    st.markdown("---")

    pagina = st.radio(
        "Navigatie",
        ["🏠 Assistent", "ℹ️ Architectuur"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.subheader("🔑 HuggingFace Token")
    if _DEFAULT_TOKEN:
        st.success("Token geconfigureerd via secrets.")
        hf_token = _DEFAULT_TOKEN
    else:
        st.markdown(
            "Haal een gratis token op via "
            "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)."
        )
        hf_token = st.text_input(
            "HF Token (optioneel)",
            type="password",
            placeholder="hf_...",
            help="Vereist voor AI-antwoorden. Zonder token worden alleen de wetsartikelen getoond.",
        )

    st.markdown("---")
    st.subheader("📊 Status")
    if st.session_state.pipeline and st.session_state.pipeline.aantal_chunks > 0:
        st.metric("Geïndexeerde chunks", st.session_state.pipeline.aantal_chunks)
        if st.session_state.geladen_wetten:
            st.markdown("**Geladen wetten:**")
            for wet in st.session_state.geladen_wetten:
                st.markdown(f"- {wet}")
    else:
        st.info("Nog geen wetten geladen.")

    st.markdown("---")
    if st.button("🗑️ Index wissen", type="secondary", use_container_width=True):
        if st.session_state.pipeline:
            st.session_state.pipeline.wis_index()
            st.session_state.geladen_wetten = []
            st.session_state.chat_history = []
            st.success("Index gewist!")
            st.rerun()


# ─────────────────────────────────────────────
# Pagina: Assistent
# ─────────────────────────────────────────────
if pagina == "🏠 Assistent":
    st.title("⚖️ Nederlandse Wetgeving Assistent")
    st.markdown(
        "Laad één of meer wetten en stel daarna vragen. "
        "De app zoekt de relevante artikelen op en genereert een antwoord."
    )

    # Sectie: Wetgeving laden
    with st.container(border=True):
        st.subheader("📚 Wetgeving laden")

        col1, col2 = st.columns([3, 1])
        with col1:
            geselecteerde_wetten = st.multiselect(
                "Selecteer wetten",
                options=list(BEKENDE_WETTEN.keys()),
                placeholder="Kies één of meer wetten...",
            )
        with col2:
            handmatig_bwbr = st.text_input(
                "Of voer een BWB ID in",
                placeholder="bijv. BWBR0001840",
            )

        laad_knop = st.button("⬇️ Laden en indexeren", type="primary", use_container_width=True)

        if laad_knop:
            te_laden = [BEKENDE_WETTEN[w] for w in geselecteerde_wetten]
            if handmatig_bwbr.strip():
                te_laden.append(handmatig_bwbr.strip())

            if not te_laden:
                st.warning("Selecteer eerst een wet of voer een BWB ID in.")
            else:
                pipeline = laad_pipeline(hf_token)
                scraper = WettenScraper(delay=0.5)
                alle_chunks = []
                voortgang_bar = st.progress(0)

                for i, bwbr_id in enumerate(te_laden):
                    voortgang_bar.progress(i / len(te_laden))
                    wet = scraper.haal_wet_op_bwbr(bwbr_id)
                    if wet:
                        chunks = plat_maken([wet])
                        alle_chunks.extend(chunks)
                        if wet["titel"] not in st.session_state.geladen_wetten:
                            st.session_state.geladen_wetten.append(wet["titel"])
                        st.success(f"✅ {wet['titel']} — {len(wet['artikelen'])} artikelen")
                    else:
                        st.error(f"❌ Kon {bwbr_id} niet ophalen")

                if alle_chunks:
                    voortgang_bar.progress(0.9)
                    pipeline.voeg_chunks_toe(alle_chunks)
                    voortgang_bar.progress(1.0)
                    st.success(f"Geïndexeerd: {pipeline.aantal_chunks} chunks totaal.")
                    st.rerun()

    # Sectie: Chat
    st.subheader("💬 Stel een vraag")

    pipeline = laad_pipeline(hf_token)

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
                        with st.expander(f"📖 Bronnen ({len(bericht['bronnen'])} artikelen)"):
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
            if st.button("🗑️ Gesprek wissen", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

# ─────────────────────────────────────────────
# Pagina: Architectuur
# ─────────────────────────────────────────────
elif pagina == "ℹ️ Architectuur":
    st.title("ℹ️ Architectuur")
    st.markdown(
        "Deze applicatie gebruikt een **RAG (Retrieval-Augmented Generation)** pipeline "
        "om vragen over Nederlandse wetgeving te beantwoorden."
    )

    st.subheader("Hoe het werkt")
    st.markdown("""
    1. **Scraper** ([scraper.py](scraper.py)) — haalt wetteksten op van
       [wetten.overheid.nl](https://wetten.overheid.nl) via officiële BWB IDs.
       De tekst wordt per artikel opgesplitst in chunks.

    2. **Embeddings** ([rag_pipeline.py](rag_pipeline.py)) — elk chunk wordt omgezet naar
       een vector met het model `paraphrase-multilingual-MiniLM-L12-v2` van
       sentence-transformers (~50 MB, meertalig, draait lokaal).

    3. **Vector store** — de vectors worden opgeslagen in een FAISS index
       (`IndexFlatIP` met genormaliseerde embeddings = cosine similarity).
       De index wordt lokaal opgeslagen in `data/`.

    4. **Retrieval** — bij een vraag wordt de vraag ook geëmbed en worden de
       *k* meest gelijkaardige artikelen opgezocht via nearest-neighbor search.

    5. **Generatie** — de gevonden artikelen worden als context meegegeven aan
       `HuggingFaceH4/zephyr-7b-beta` via de HuggingFace Inference API
       (gratis tier, `provider="hf-inference"`). Het model genereert een
       Nederlands antwoord met verwijzing naar de relevante artikelen.
    """)

    st.subheader("Componentenoverzicht")
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

    st.subheader("Diagram")
    st.code("""
┌─────────────────────────────────────┐
│           Streamlit UI (app.py)     │
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

    st.subheader("Beperkingen")
    st.markdown("""
    - De FAISS index wordt **niet persistent opgeslagen** op Streamlit Community Cloud —
      gebruikers moeten wetten laden bij elk nieuw sessie.
    - De gratis HuggingFace Inference API heeft **rate limits** bij intensief gebruik.
    - Wetteksten worden **niet automatisch bijgewerkt** — wis de index en herlaad
      voor de meest actuele versie.
    """)
