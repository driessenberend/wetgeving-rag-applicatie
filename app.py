"""
Streamlit RAG-applicatie voor Nederlandse wetgeving.
Stel vragen over Nederlandse wetten via een gratis AI-assistent.
"""

import streamlit as st
from pathlib import Path

from scraper import WettenScraper, BEKENDE_WETTEN, plat_maken
from rag_pipeline import RAGPipeline, BESCHIKBARE_MODELLEN

# HF token via Streamlit secrets (voor Streamlit Cloud deployment)
_DEFAULT_TOKEN = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""

# ─────────────────────────────────────────────
# Pagina-configuratie
# ─────────────────────────────────────────────
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


def laad_pipeline(hf_token: str, model_naam: str) -> RAGPipeline:
    """Initialiseer of hergebruik de RAG pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("Embedding model laden (eenmalig ~30 seconden)..."):
            pipeline = RAGPipeline(
                hf_token=hf_token,
                model_naam=model_naam,
                data_map="data",
            )
            # Probeer bestaande index te laden
            if pipeline.laden():
                st.success(f"Bestaande index geladen ({pipeline.aantal_chunks} chunks)")
            st.session_state.pipeline = pipeline
    else:
        # Update LLM instellingen indien gewijzigd
        if hf_token:
            st.session_state.pipeline.stel_llm_in(model_naam, hf_token)
    return st.session_state.pipeline


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Wetgeving RAG")
    st.markdown("---")

    st.subheader("🔑 HuggingFace Token")
    st.markdown(
        "Haal een gratis token op via [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
        "(gratis account vereist)."
    )
    hf_token = st.text_input(
        "HF Token (optioneel)",
        value=_DEFAULT_TOKEN,
        type="password",
        placeholder="hf_...",
        help="Vereist voor AI-antwoorden. Zonder token worden alleen de wetsartikelen getoond.",
    )

    st.subheader("🤖 Taalmodel")
    model_keuze = st.selectbox(
        "Kies een model",
        options=list(BESCHIKBARE_MODELLEN.keys()),
        index=0,
        help="Mistral 7B werkt het beste voor Nederlands.",
    )
    gekozen_model = BESCHIKBARE_MODELLEN[model_keuze]

    st.markdown("---")
    st.subheader("📊 Status")
    if st.session_state.pipeline:
        n = st.session_state.pipeline.aantal_chunks
        st.metric("Geïndexeerde chunks", n)
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

    st.markdown("---")
    st.markdown(
        "**Bronnen:** [wetten.overheid.nl](https://wetten.overheid.nl) | "
        "[HuggingFace](https://huggingface.co) | "
        "[GitHub](https://github.com)"
    )


# ─────────────────────────────────────────────
# Hoofd-inhoud
# ─────────────────────────────────────────────
st.title("⚖️ Nederlandse Wetgeving RAG Assistent")
st.markdown(
    "Stel vragen over Nederlandse wetten. De app haalt wetteksten op van "
    "[wetten.overheid.nl](https://wetten.overheid.nl) en beantwoordt uw vragen "
    "op basis van de actuele wetgeving."
)

tab1, tab2, tab3 = st.tabs(["📚 Wetgeving laden", "💬 Vragen stellen", "ℹ️ Over"])

# ─────────────────────────────────────────────
# Tab 1: Wetgeving laden
# ─────────────────────────────────────────────
with tab1:
    st.header("Wetgeving laden en indexeren")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bekende wetten")
        st.markdown("Selecteer een of meerdere standaard wetten:")

        geselecteerde_wetten = st.multiselect(
            "Kies wetten",
            options=list(BEKENDE_WETTEN.keys()),
            default=[],
            placeholder="Selecteer wetten...",
        )

        if geselecteerde_wetten:
            st.info(f"{len(geselecteerde_wetten)} wet(ten) geselecteerd")

    with col2:
        st.subheader("Zoeken op wetten.overheid.nl")
        st.markdown("Zoek en laad een specifieke wet:")

        zoekterm = st.text_input(
            "Zoekterm",
            placeholder="bijv. arbeidsrecht, belasting, huurrecht...",
        )
        max_zoekresultaten = st.slider("Max. zoekresultaten", 1, 10, 3)

        zoek_knop = st.button("🔍 Zoeken", type="secondary")

        if zoek_knop and zoekterm:
            with st.spinner(f"Zoeken naar '{zoekterm}'..."):
                scraper = WettenScraper()
                zoekresultaten = scraper.zoek_wetten(zoekterm, max_resultaten=max_zoekresultaten)

            if zoekresultaten:
                st.success(f"{len(zoekresultaten)} resultaten gevonden")
                gevonden_selectie = []
                for r in zoekresultaten:
                    aangevinkt = st.checkbox(
                        f"**{r['titel']}** ({r['bwbr_id']})",
                        key=f"zoek_{r['bwbr_id']}",
                    )
                    if aangevinkt:
                        gevonden_selectie.append(r["bwbr_id"])
            else:
                st.warning(
                    "Geen resultaten gevonden. Probeer een andere zoekterm "
                    "of gebruik de bekende wetten links."
                )

    st.markdown("---")

    # BWB ID handmatig invoeren
    st.subheader("Wet laden via BWB ID")
    col3, col4 = st.columns([3, 1])
    with col3:
        handmatig_bwbr = st.text_input(
            "BWB ID",
            placeholder="bijv. BWBR0001840 (Grondwet)",
            help="Te vinden in de URL op wetten.overheid.nl",
        )
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        voeg_toe_knop = st.button("Toevoegen", type="secondary")

    # Laad-knop
    st.markdown("---")
    laad_knop = st.button("⬇️ Wetten laden en indexeren", type="primary", use_container_width=True)

    if laad_knop:
        # Verzamel alle te laden BWBR IDs
        te_laden = []
        for wet_naam in geselecteerde_wetten:
            te_laden.append(BEKENDE_WETTEN[wet_naam])

        if handmatig_bwbr and voeg_toe_knop:
            te_laden.append(handmatig_bwbr.strip())

        # Ook handmatig ID als laad-knop direct geklikt wordt
        if handmatig_bwbr and not voeg_toe_knop:
            te_laden.append(handmatig_bwbr.strip())

        if not te_laden:
            st.warning("Selecteer eerst één of meer wetten of voer een BWB ID in.")
        else:
            # Initialiseer pipeline
            pipeline = laad_pipeline(hf_token, gekozen_model)

            voortgang_bar = st.progress(0)
            status_tekst = st.empty()

            scraper = WettenScraper(delay=0.5)
            alle_chunks = []

            for i, bwbr_id in enumerate(te_laden):
                voortgang = i / len(te_laden)
                voortgang_bar.progress(voortgang)
                status_tekst.text(f"Ophalen: {bwbr_id} ({i+1}/{len(te_laden)})...")

                wet = scraper.haal_wet_op_bwbr(bwbr_id)
                if wet:
                    chunks = plat_maken([wet])
                    alle_chunks.extend(chunks)
                    titel = wet["titel"]
                    if titel not in st.session_state.geladen_wetten:
                        st.session_state.geladen_wetten.append(titel)
                    st.success(
                        f"✅ {titel}: {len(wet['artikelen'])} artikelen opgehaald"
                    )
                else:
                    st.error(f"❌ Kon {bwbr_id} niet ophalen")

            if alle_chunks:
                voortgang_bar.progress(0.8)
                status_tekst.text(f"Indexeren van {len(alle_chunks)} chunks...")

                n_geindexeerd = pipeline.voeg_chunks_toe(alle_chunks)

                voortgang_bar.progress(1.0)
                status_tekst.text("")
                st.success(
                    f"🎉 Klaar! {n_geindexeerd} chunks geïndexeerd. "
                    f"Totaal in index: {pipeline.aantal_chunks} chunks."
                )
                st.info("Ga naar het tabblad **💬 Vragen stellen** om vragen te stellen.")
            else:
                st.error("Geen tekst kunnen ophalen. Controleer de BWB IDs en probeer opnieuw.")

# ─────────────────────────────────────────────
# Tab 2: Vragen stellen
# ─────────────────────────────────────────────
with tab2:
    st.header("Stel een vraag over de wetgeving")

    # Initialiseer pipeline als dat nog niet gebeurd is
    pipeline = laad_pipeline(hf_token, gekozen_model)

    if pipeline.aantal_chunks == 0:
        st.warning(
            "⚠️ Er zijn nog geen wetten geladen. "
            "Ga naar **📚 Wetgeving laden** om wetten te importeren."
        )
    else:
        st.info(f"Beschikbare kennis: {pipeline.aantal_chunks} wetsartikelen/-passages geïndexeerd.")

    # Instellingen
    with st.expander("⚙️ Instellingen", expanded=False):
        k_chunks = st.slider(
            "Aantal te raadplegen artikelen (k)",
            min_value=1, max_value=10, value=5,
            help="Meer artikelen = rijkere context, maar trager",
        )
        max_tokens = st.slider(
            "Max. lengte antwoord (tokens)",
            min_value=128, max_value=1024, value=512, step=64,
        )
        toon_bronnen = st.checkbox("Bronnen tonen", value=True)

    # Voorbeeldvragen
    st.markdown("**Voorbeeldvragen:**")
    voorbeeld_vragen = [
        "Wat zijn de opzegtermijnen bij een arbeidsovereenkomst?",
        "Wanneer heeft een werknemer recht op transitievergoeding?",
        "Wat is de wettelijke proeftijd bij een vast contract?",
        "Hoe werkt de ontslagbescherming bij ziekte?",
        "Wat zijn de rechten van een huurder bij gebreken?",
    ]
    cols = st.columns(len(voorbeeld_vragen))
    for col, vraag in zip(cols, voorbeeld_vragen):
        if col.button(vraag[:40] + "...", key=f"voorbeeld_{vraag[:20]}", use_container_width=True):
            st.session_state["actieve_vraag"] = vraag

    # Chat invoer
    vraag_input = st.chat_input(
        "Stel uw vraag over de Nederlandse wetgeving...",
        disabled=(pipeline.aantal_chunks == 0),
    )

    # Gebruik voorbeeldvraag of chat input
    actieve_vraag = st.session_state.pop("actieve_vraag", None) or vraag_input

    if actieve_vraag:
        # Voeg vraag toe aan history
        st.session_state.chat_history.append({"rol": "gebruiker", "inhoud": actieve_vraag})

        with st.spinner("Antwoord genereren..."):
            resultaat = pipeline.stel_vraag(
                actieve_vraag,
                k=k_chunks,
                max_tokens=max_tokens,
            )

        # Voeg antwoord toe aan history
        st.session_state.chat_history.append({
            "rol": "assistent",
            "inhoud": resultaat["antwoord"],
            "bronnen": resultaat.get("bronnen", []),
        })

    # Toon chat history
    if st.session_state.chat_history:
        st.markdown("---")
        for bericht in st.session_state.chat_history:
            if bericht["rol"] == "gebruiker":
                with st.chat_message("user"):
                    st.markdown(bericht["inhoud"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(bericht["inhoud"])

                    # Toon bronnen
                    if toon_bronnen and bericht.get("bronnen"):
                        with st.expander(f"📖 Bronnen ({len(bericht['bronnen'])} artikelen)", expanded=False):
                            for bron in bericht["bronnen"]:
                                relevantie = int(bron["score"] * 100)
                                st.markdown(
                                    f"**{bron['titel']} – {bron['artikel']}** "
                                    f"*(relevantie: {relevantie}%)*"
                                )
                                if bron.get("url"):
                                    st.markdown(f"[Bekijk op wetten.overheid.nl]({bron['url']})")
                                st.markdown(f"> {bron['tekst']}...")
                                st.markdown("---")

        # Geschiedenis wissen
        if st.button("🗑️ Gesprek wissen", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown(
            "👆 Stel een vraag hierboven of klik op een voorbeeldvraag om te beginnen."
        )

# ─────────────────────────────────────────────
# Tab 3: Over
# ─────────────────────────────────────────────
with tab3:
    st.header("Over deze applicatie")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏗️ Architectuur")
        st.markdown("""
        Deze applicatie gebruikt een **RAG (Retrieval-Augmented Generation)** pipeline:

        1. **Scraper** — Haalt wetteksten op van [wetten.overheid.nl](https://wetten.overheid.nl)
        2. **Embeddings** — `sentence-transformers` met het meertalige model
           `paraphrase-multilingual-MiniLM-L12-v2`
        3. **Vector store** — FAISS (Facebook AI Similarity Search) voor snelle
           cosine-similarity zoekopdrachten
        4. **LLM** — HuggingFace Inference API (gratis tier) met modellen als
           Mistral 7B of Zephyr 7B
        5. **Frontend** — Streamlit

        **Volledig gratis:** geen betaalde API keys vereist!
        """)

    with col2:
        st.subheader("🚀 Snel starten")
        st.markdown("""
        **Stap 1:** Haal een gratis HuggingFace token op
        → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

        **Stap 2:** Voer het token in de sidebar in

        **Stap 3:** Ga naar **📚 Wetgeving laden** en selecteer wetten

        **Stap 4:** Stel vragen via **💬 Vragen stellen**

        ---
        **Zonder token** werkt de app ook: dan worden de meest relevante
        wetsartikelen getoond zonder AI-samenvatting.
        """)

    st.subheader("📋 Veelgestelde vragen")

    with st.expander("Werkt dit offline?"):
        st.markdown(
            "De embeddings worden lokaal berekend (geen internet nodig na het laden van het model). "
            "Voor AI-antwoorden is de HuggingFace Inference API vereist, wat internet nodig heeft. "
            "Het scrapen vereist uiteraard ook internet."
        )

    with st.expander("Is de wetgeving actueel?"):
        st.markdown(
            "De app haalt telkens de huidige versie van de wet op van wetten.overheid.nl. "
            "De index wordt lokaal opgeslagen en wordt niet automatisch bijgewerkt. "
            "Wis de index en laad de wetten opnieuw voor de meest actuele versie."
        )

    with st.expander("Wat zijn de beperkingen van de gratis HF tier?"):
        st.markdown(
            "De gratis HuggingFace Inference API heeft rate limits. Bij veel gebruik kan "
            "de API tijdelijk vertragen. Grote modellen (7B+) kunnen langzamer zijn op de "
            "gratis tier. Overweeg de [HF PRO abonnement](https://huggingface.co/pricing) "
            "voor intensief gebruik."
        )

    with st.expander("Kan ik ook andere wetten toevoegen?"):
        st.markdown(
            "Ja! Elke wet op wetten.overheid.nl heeft een BWB ID (te vinden in de URL). "
            "Voer dit ID in onder **Wet laden via BWB ID**. "
            "Voorbeeld: `BWBR0001840` voor de Grondwet."
        )

    st.markdown("---")
    st.markdown(
        "Gebouwd met ❤️ | Data: [wetten.overheid.nl](https://wetten.overheid.nl) (Overheid.nl, CC0) | "
        "Embeddings: [sentence-transformers](https://www.sbert.net/) | "
        "Vector store: [FAISS](https://faiss.ai/) | "
        "LLM: [HuggingFace](https://huggingface.co/)"
    )
