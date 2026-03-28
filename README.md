# ⚖️ Nederlandse Wetgeving RAG Assistent

Stel vragen over Nederlandse wetten via een gratis AI-assistent. De app haalt wetteksten op van [wetten.overheid.nl](https://wetten.overheid.nl), indexeert ze in een lokale vector store, en beantwoordt vragen in het Nederlands.

**Volledig gratis** — geen betaalde API keys nodig.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Functionaliteiten

- **Scraper** — Haalt actuele wetteksten op van wetten.overheid.nl via BWB IDs
- **Embeddings** — Meertalig sentence-transformers model (werkt goed voor Nederlands)
- **Vector store** — FAISS voor snelle similarity search
- **LLM** — HuggingFace Inference API (gratis tier), o.a. Mistral 7B
- **UI** — Streamlit met chat interface en bronvermelding

## Installatie

### Vereisten

- Python 3.9+
- pip

### Lokaal draaien

```bash
# Clone of download het project
cd wetgeving-rag

# Installeer dependencies
pip install -r requirements.txt

# Start de app
streamlit run app.py
```

De app is bereikbaar op `http://localhost:8501`.

### HuggingFace token (gratis)

Voor AI-gegenereerde antwoorden heb je een gratis HuggingFace token nodig:

1. Maak een gratis account op [huggingface.co](https://huggingface.co)
2. Ga naar [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Maak een token aan met **Read** rechten
4. Voer het token in de sidebar van de app in

**Zonder token** toont de app de meest relevante wetsartikelen zonder AI-samenvatting.

## Gebruik

### Stap 1 — Wetgeving laden

Ga naar het tabblad **📚 Wetgeving laden**:

- Selecteer één of meer wetten uit de keuzelijst (20+ veelgebruikte wetten beschikbaar)
- Of zoek op trefwoord (bijv. "arbeidsrecht")
- Of voer een BWB ID handmatig in (te vinden in de URL op wetten.overheid.nl)
- Klik op **Wetten laden en indexeren**

### Stap 2 — Vragen stellen

Ga naar het tabblad **💬 Vragen stellen** en typ uw vraag.

**Voorbeeldvragen:**
- "Wat zijn de opzegtermijnen bij een arbeidsovereenkomst?"
- "Wanneer heeft een werknemer recht op transitievergoeding?"
- "Wat zijn de rechten van een huurder bij gebreken?"
- "Welke straffen staan er op oplichting?"

## Deployment op Streamlit Community Cloud

1. Push het project naar een publieke of private GitHub repository
2. Ga naar [share.streamlit.io](https://share.streamlit.io) en log in
3. Maak een nieuwe app aan en selecteer `app.py`
4. Voeg je HuggingFace token toe als secret:
   - Ga naar App Settings → Secrets
   - Voeg toe: `HF_TOKEN = "hf_jouw_token"`
5. Deploy!

> **Let op:** De vector index wordt niet mee-gedeployed. Gebruikers moeten wetten laden bij eerste gebruik.

## Architectuur

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                  │
└─────────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
┌────────▼──────────┐     ┌──────────▼──────────┐
│  scraper.py        │     │  rag_pipeline.py     │
│  wetten.overheid.nl│     │                     │
│  HTML → artikelen  │     │  ┌───────────────┐  │
└────────────────────┘     │  │EmbeddingModel │  │
                           │  │sentence-transf│  │
                           │  └───────┬───────┘  │
                           │          │           │
                           │  ┌───────▼───────┐  │
                           │  │  VectorStore  │  │
                           │  │  FAISS index  │  │
                           │  └───────┬───────┘  │
                           │          │           │
                           │  ┌───────▼───────┐  │
                           │  │   LLMClient   │  │
                           │  │  HF Inference │  │
                           │  └───────────────┘  │
                           └─────────────────────┘
```

## Beschikbare wetten

De app ondersteunt alle wetten op wetten.overheid.nl. Standaard beschikbaar:

| Wet | BWB ID |
|-----|--------|
| Grondwet | BWBR0001840 |
| BW Boek 7 (Arbeidsrecht) | BWBR0005290 |
| Wetboek van Strafrecht | BWBR0001854 |
| Algemene wet bestuursrecht | BWBR0005537 |
| Wet minimumloon | BWBR0002638 |
| Wet op de CAO | BWBR0001841 |
| ... en 15+ meer | |

## Technische details

| Component | Technologie | Reden |
|-----------|-------------|-------|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Klein (~50MB), goed voor NL |
| Vector store | FAISS (IndexFlatIP) | Snel, gratis, lokaal |
| Similarity | Cosine (via genormaliseerde embeddings) | Beste voor semantisch zoeken |
| LLM | Mistral 7B via HF Inference API | Gratis, goed voor NL |
| Chunking | Per wetsartikel + fallback (1000 tekens) | Juridisch zinvolle eenheden |

## Licentie

De code is beschikbaar onder de MIT licentie. De wetteksten zijn afkomstig van wetten.overheid.nl en zijn gepubliceerd onder een open licentie (CC0 / overheidswerk).
