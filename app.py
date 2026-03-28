"""
Streamlit RAG-applicatie voor Nederlandse wetgeving.
"""

import streamlit as st

from scraper import WettenScraper, BEKENDE_WETTEN, plat_maken

_HF_TOKEN = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""

st.set_page_config(
    page_title="Wetgeving Assistent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state ──────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "geladen_wetten" not in st.session_state:
    st.session_state.geladen_wetten = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_results_prev" not in st.session_state:
    st.session_state.eval_results_prev = None
if "eval_summary" not in st.session_state:
    st.session_state.eval_summary = None
if "eval_summary_prev" not in st.session_state:
    st.session_state.eval_summary_prev = None
if "eval_judge_model" not in st.session_state:
    st.session_state.eval_judge_model = None
if "eval_judge_model_prev" not in st.session_state:
    st.session_state.eval_judge_model_prev = None
if "eval_llm_model" not in st.session_state:
    st.session_state.eval_llm_model = "Qwen/Qwen2.5-7B-Instruct"
if "eval_embedding_model" not in st.session_state:
    st.session_state.eval_embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
if "eval_top_k" not in st.session_state:
    st.session_state.eval_top_k = 5
if "eval_mode" not in st.session_state:
    st.session_state.eval_mode = "snel"


def laad_pipeline():
    if st.session_state.pipeline is None:
        from rag_pipeline import RAGPipeline
        with st.spinner("Model laden..."):
            pipeline = RAGPipeline(hf_token=_HF_TOKEN, data_map="data")
            pipeline.laden()
            st.session_state.pipeline = pipeline
    return st.session_state.pipeline


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Wetgeving Assistent")
    st.markdown("---")

    pagina = st.radio(
        "Navigatie",
        ["Assistent", "Evaluatie", "Architectuur"],
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

    # Evaluation config (only shown on Evaluatie page)
    if pagina == "Evaluatie":
        st.markdown("---")
        st.markdown("**Evaluatie-instellingen**")

        from evaluation import EVAL_MODELLEN, EMBEDDING_MODELLEN

        st.session_state.eval_llm_model = st.selectbox(
            "Beoordelingsmodel",
            options=EVAL_MODELLEN,
            index=EVAL_MODELLEN.index(st.session_state.eval_llm_model)
                  if st.session_state.eval_llm_model in EVAL_MODELLEN else 0,
        )

        embedding_labels = list(EMBEDDING_MODELLEN.keys())
        embedding_values = list(EMBEDDING_MODELLEN.values())
        current_emb = st.session_state.eval_embedding_model
        current_emb_idx = (
            embedding_values.index(current_emb)
            if current_emb in embedding_values else 0
        )
        selected_emb_label = st.selectbox(
            "Embeddingmodel",
            options=embedding_labels,
            index=current_emb_idx,
            help="Bij wijziging wordt de index opnieuw opgebouwd.",
        )
        st.session_state.eval_embedding_model = EMBEDDING_MODELLEN[selected_emb_label]

        st.session_state.eval_top_k = st.slider(
            "Top-K documenten",
            min_value=1, max_value=10,
            value=st.session_state.eval_top_k,
        )

        modus_opties = ["Snel (5 vragen)", "Volledig (15 vragen)"]
        modus_idx = 0 if st.session_state.eval_mode == "snel" else 1
        gekozen_modus = st.radio("Evaluatiemodus", modus_opties, index=modus_idx)
        st.session_state.eval_mode = "snel" if gekozen_modus.startswith("Snel") else "volledig"


# ─── Pagina: Assistent ───────────────────────────────────────────────────────
if pagina == "Assistent":
    st.markdown("## Nederlandse Wetgeving Assistent")
    st.markdown(
        "Laad een of meer wetten en stel vervolgens vragen. "
        "De assistent zoekt de relevante artikelen op en genereert een antwoord."
    )
    st.markdown("---")

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
    st.markdown("### Vragen stellen")

    pipeline = st.session_state.pipeline
    if pipeline is None or pipeline.aantal_chunks == 0:
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


# ─── Pagina: Evaluatie ───────────────────────────────────────────────────────
elif pagina == "Evaluatie":
    import pandas as pd
    from evaluation import Evaluator, load_test_cases

    st.markdown("## Evaluatie")
    st.markdown(
        "Meet de kwaliteit van de RAG-pipeline op een vaste set juridische testvragen. "
        "De pipeline genereert antwoorden; een LLM beoordeelt vervolgens de kwaliteit."
    )
    st.markdown("---")

    pipeline = st.session_state.pipeline
    if pipeline is None or pipeline.aantal_chunks == 0:
        st.warning(
            "Laad eerst wetgeving via de pagina **Assistent** voordat u de evaluatie uitvoert. "
            "De testvragen hebben de volgende wetten nodig: "
            "Grondwet, BW Boek 7, Wetboek van Strafrecht, Awb, Wet minimumloon, Arbeidsomstandighedenwet."
        )
    else:
        if not _HF_TOKEN:
            st.warning("Geen HuggingFace token gevonden. Voeg HF_TOKEN toe aan de Streamlit secrets.")

        if st.button("Evaluatie uitvoeren", type="primary", disabled=not _HF_TOKEN):
            # Rebuild index if embedding model changed
            current_emb = st.session_state.eval_embedding_model
            pipeline = laad_pipeline()
            if current_emb != "paraphrase-multilingual-MiniLM-L12-v2":
                with st.spinner(f"Index opnieuw opbouwen met {current_emb}..."):
                    try:
                        pipeline.rebuild_index(current_emb)
                    except Exception as e:
                        st.error(f"Index herbouwen mislukt: {e}")
                        st.stop()

            # Archive previous results
            if st.session_state.eval_results:
                st.session_state.eval_results_prev = st.session_state.eval_results
                st.session_state.eval_summary_prev = st.session_state.eval_summary
                st.session_state.eval_judge_model_prev = st.session_state.eval_judge_model

            test_cases = load_test_cases("test_cases.json")
            evaluator = Evaluator(
                hf_token=_HF_TOKEN,
                judge_model=st.session_state.eval_llm_model,
            )

            voortgang_bar = st.progress(0)
            status = st.empty()

            def progress_callback(current, total, message):
                voortgang_bar.progress(current / max(total, 1))
                status.markdown(message)

            with st.spinner("Evaluatie bezig..."):
                results = evaluator.run_evaluation(
                    pipeline=pipeline,
                    test_cases=test_cases,
                    mode=st.session_state.eval_mode,
                    top_k=st.session_state.eval_top_k,
                    progress_callback=progress_callback,
                )

            st.session_state.eval_results = results
            st.session_state.eval_summary = evaluator.compute_summary(results)
            st.session_state.eval_judge_model = st.session_state.eval_llm_model
            voortgang_bar.progress(1.0)
            status.empty()
            st.rerun()

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.eval_results:
        summary = st.session_state.eval_summary
        prev_summary = st.session_state.eval_summary_prev

        st.markdown("### Resultaten")

        if st.session_state.eval_judge_model:
            st.caption(f"Scores gegeven door: `{st.session_state.eval_judge_model}`")

        # Metric cards
        col_names = ["Correctheid", "Trouw", "Volledigheid", "Helderheid", "Relevantie"]
        metric_keys = ["token_overlap", "trouw", "volledigheid", "helderheid", "relevantie"]
        metric_help = {
            "Correctheid": "Mate van overlap tussen het gegenereerde antwoord en het referentieantwoord op woordniveau (F1-score, 0–1).",
            "Trouw": "Bevat het antwoord alleen beweringen die in de opgehaalde wetteksten staan? Detecteert hallucinaties. (1–5)",
            "Volledigheid": "Dekt het antwoord de kernpunten van het verwachte antwoord? (1–5)",
            "Helderheid": "Is het antwoord duidelijk en goed gestructureerd? (1–5)",
            "Relevantie": "Beantwoordt het antwoord de gestelde vraag? (1–5)",
        }

        cols = st.columns(5)
        for col, name, key in zip(cols, col_names, metric_keys):
            val = summary.get(key)
            prev_val = prev_summary.get(key) if prev_summary else None
            if val is None:
                display = "N/B"
                delta = None
            elif key == "token_overlap":
                display = f"{val:.0%}"
                delta = f"{(val - prev_val):.0%}" if prev_val is not None else None
            else:
                display = f"{val:.1f} / 5"
                delta = f"{(val - prev_val):+.2f}" if prev_val is not None else None
            col.metric(name, display, delta=delta, help=metric_help[name])

        st.markdown("---")

        # Detail table
        st.markdown("### Detail per vraag")
        rows = []
        for r in st.session_state.eval_results:
            rows.append({
                "ID": r.id,
                "Vraag": r.vraag[:70] + ("..." if len(r.vraag) > 70 else ""),
                "Correctheid": r.token_overlap,
                "Trouw": r.trouw,
                "Volledigheid": r.volledigheid,
                "Helderheid": r.helderheid,
                "Relevantie": r.relevantie,
                "Latentie (ms)": int(r.latency_ms),
            })
        df = pd.DataFrame(rows)

        def _color_score(val, vmax=5):
            """Return a CSS background-color based on score value."""
            if val is None:
                return "background-color: #f0f0f0; color: #999"
            ratio = float(val) / vmax
            if ratio >= 0.7:
                return "background-color: #c6efce; color: #276221"
            elif ratio >= 0.4:
                return "background-color: #ffeb9c; color: #9c6500"
            else:
                return "background-color: #ffc7ce; color: #9c0006"

        styled = (
            df.style
            .map(_color_score, subset=["Trouw", "Volledigheid", "Helderheid", "Relevantie"], vmax=5)
            .map(_color_score, subset=["Correctheid"], vmax=1)
            .format({
                "Correctheid": lambda v: f"{v:.0%}" if v is not None else "N/B",
                "Trouw": lambda v: str(v) if v is not None else "N/B",
                "Volledigheid": lambda v: str(v) if v is not None else "N/B",
                "Helderheid": lambda v: str(v) if v is not None else "N/B",
                "Relevantie": lambda v: str(v) if v is not None else "N/B",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Per-question expandable details
        st.markdown("### Details per vraag")
        for r in st.session_state.eval_results:
            with st.expander(f"{r.id} — {r.vraag[:80]}"):
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("**Vraag**")
                    st.markdown(r.vraag)
                    st.markdown("**Verwacht antwoord**")
                    st.markdown(r.verwacht_antwoord)
                with col_r:
                    st.markdown("**Gegenereerd antwoord**")
                    st.markdown(r.gegeven_antwoord)
                    st.markdown(f"*Latentie: {r.latency_ms:.0f} ms*")

                st.markdown("**Scores**")
                score_cols_detail = st.columns(5)
                score_data = [
                    ("Correctheid", r.token_overlap, 1.0, True),
                    ("Trouw", r.trouw, 5, False),
                    ("Volledigheid", r.volledigheid, 5, False),
                    ("Helderheid", r.helderheid, 5, False),
                    ("Relevantie", r.relevantie, 5, False),
                ]
                for sc, (label, val, maximum, is_ratio) in zip(score_cols_detail, score_data):
                    if val is not None:
                        sc.metric(label, f"{val:.0%}" if is_ratio else f"{val}/5")
                    else:
                        sc.metric(label, "N/B")

                if r.retrieved_chunks:
                    with st.expander("Opgehaalde bronnen"):
                        for chunk, score in r.retrieved_chunks:
                            st.markdown(
                                f"**{chunk.get('titel', '')} – {chunk.get('artikel', '')}** "
                                f"*(score: {score:.2f})*"
                            )
                            st.markdown(f"> {chunk.get('tekst', '')[:300]}...")
                            st.markdown("---")

        # Comparison with previous run
        if st.session_state.eval_results_prev and st.session_state.eval_summary_prev:
            st.markdown("---")
            st.markdown("### Vergelijking met vorige run")
            prev_judge = st.session_state.eval_judge_model_prev
            curr_judge = st.session_state.eval_judge_model
            if prev_judge or curr_judge:
                st.caption(
                    f"Vorige run: `{prev_judge or 'onbekend'}` — "
                    f"Huidige run: `{curr_judge or 'onbekend'}`"
                )
            prev = st.session_state.eval_summary_prev
            curr = summary
            comp_rows = []
            for name, key in zip(col_names, metric_keys):
                v_prev = prev.get(key)
                v_curr = curr.get(key)
                if v_prev is not None and v_curr is not None:
                    delta = v_curr - v_prev
                    comp_rows.append({
                        "Metric": name,
                        "Vorige run": f"{v_prev:.0%}" if key == "token_overlap" else f"{v_prev:.2f}",
                        "Huidige run": f"{v_curr:.0%}" if key == "token_overlap" else f"{v_curr:.2f}",
                        "Verschil": f"{'+'if delta >= 0 else ''}{delta:.0%}" if key == "token_overlap"
                                    else f"{'+'if delta >= 0 else ''}{delta:.2f}",
                    })
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ─── Pagina: Architectuur ────────────────────────────────────────────────────
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
   `Qwen/Qwen2.5-7B-Instruct` via de HuggingFace Inference API. Het model genereert een
   Nederlands antwoord met verwijzing naar de relevante artikelen.

6. **Evaluatie** — een tweede LLM-aanroep beoordeelt het antwoord op vier criteria
   (trouw, volledigheid, helderheid, relevantie). Lokale tokenoverlap meet de feitelijke correctheid.
    """)

    st.markdown("### Componentenoverzicht")
    st.markdown("""
| Component | Technologie |
|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| Vector store | FAISS `IndexFlatIP` |
| Similarity | Cosine (via genormaliseerde embeddings) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API |
| Chunking | Per wetsartikel, fallback per 1000 tekens |
| Evaluatie | LLM-as-judge + token overlap F1 |
| Frontend | Streamlit |
    """)

    st.markdown("### Diagram")
    st.code("""
┌─────────────────────────────────────┐
│         Streamlit UI (app.py)       │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────────────────────────────┐
    │                                     │
┌───▼────────────┐    ┌───────────────────▼──────┐
│  scraper.py    │    │  rag_pipeline.py          │
│  HTML → chunks │    │  EmbeddingModel (lokaal)  │
└────────────────┘    │  VectorStore (FAISS)      │
                      │  LLMClient (HF API)       │
                      └───────────┬───────────────┘
                                  │
                      ┌───────────▼───────────────┐
                      │  evaluation.py            │
                      │  TokenOverlapScorer       │
                      │  LLMJudge (HF API)        │
                      └──────────────────────────┘
    """, language=None)

    st.markdown("### Beperkingen")
    st.markdown("""
- De FAISS index wordt niet persistent opgeslagen op Streamlit Community Cloud.
  Gebruikers moeten wetten laden bij elke nieuwe sessie.
- De gratis HuggingFace Inference API heeft rate limits bij intensief gebruik.
- Wetteksten worden niet automatisch bijgewerkt. Wis de index en herlaad
  voor de meest actuele versie.
- De evaluatiemodule vereist een geladen index en een HuggingFace token.
    """)
