"""
Scraper voor wetten.overheid.nl
Haalt wetteksten op via HTML scraping en de officiële BWBR IDs.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Veelgebruikte wetten met hun BWB IDs
BEKENDE_WETTEN = {
    "Grondwet": "BWBR0001840",
    "Burgerlijk Wetboek Boek 1 (Personen- en familierecht)": "BWBR0002656",
    "Burgerlijk Wetboek Boek 2 (Rechtspersonen)": "BWBR0003045",
    "Burgerlijk Wetboek Boek 6 (Verbintenissenrecht)": "BWBR0005289",
    "Burgerlijk Wetboek Boek 7 (Bijzondere overeenkomsten / Arbeidsrecht)": "BWBR0005290",
    "Wetboek van Strafrecht": "BWBR0001854",
    "Wetboek van Burgerlijke Rechtsvordering": "BWBR0001827",
    "Algemene wet bestuursrecht": "BWBR0005537",
    "Wet minimumloon en minimumvakantiebijslag": "BWBR0002638",
    "Arbeidsomstandighedenwet": "BWBR0010346",
    "Wet op de collectieve arbeidsovereenkomst": "BWBR0001841",
    "Wet werk en zekerheid (Flexwet)": "BWBR0006502",
    "Wet op het financieel toezicht": "BWBR0020368",
    "Algemene wet inzake rijksbelastingen": "BWBR0002320",
    "Wet inkomstenbelasting 2001": "BWBR0011353",
    "Wet op de omzetbelasting 1968": "BWBR0002629",
    "Vreemdelingenwet 2000": "BWBR0011823",
    "Wet bescherming persoonsgegevens (oud)": "BWBR0011468",
    "Uitvoeringswet AVG": "BWBR0040940",
    "Opiumwet": "BWBR0001941",
}

BASE_URL = "https://wetten.overheid.nl"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class WettenScraper:
    def __init__(self, delay: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        try:
            time.sleep(self.delay)
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Fout bij ophalen {url}: {e}")
            return None

    def zoek_wetten(self, zoekterm: str, max_resultaten: int = 5) -> List[Dict]:
        """
        Zoekt naar wetten op wetten.overheid.nl.
        Geeft lijst terug met {'titel', 'bwbr_id', 'url'}.
        """
        url = f"{BASE_URL}/zoeken"
        params = {
            "q": zoekterm,
            "rows": max_resultaten,
            "type": "wet",
        }
        response = self._get(url, params=params)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "lxml")
        resultaten = []

        # Zoekresultaten staan in een lijst
        result_items = soup.select("ul.results li") or soup.select(".search-result")
        if not result_items:
            # Probeer alternatieve selectors
            result_items = soup.select("article") or soup.select(".result-item")

        for item in result_items[:max_resultaten]:
            link_tag = item.find("a", href=True)
            if not link_tag:
                continue

            href = link_tag.get("href", "")
            # BWB IDs zitten in de URL: /BWBR0001840/
            bwbr_match = re.search(r"/(BWBR\d+)/", href)
            if not bwbr_match:
                continue

            bwbr_id = bwbr_match.group(1)
            titel = link_tag.get_text(strip=True)
            resultaten.append({
                "titel": titel,
                "bwbr_id": bwbr_id,
                "url": urljoin(BASE_URL, href),
            })

        return resultaten

    def haal_wet_op_bwbr(self, bwbr_id: str) -> Optional[Dict]:
        """
        Haalt een wet op via het BWBR ID (bijv. 'BWBR0001840').
        Geeft dict terug met {'titel', 'bwbr_id', 'url', 'artikelen'}.
        """
        url = f"{BASE_URL}/{bwbr_id}/"
        response = self._get(url)
        if not response:
            return None

        # Volg eventuele redirects naar de actuele versie
        final_url = response.url
        soup = BeautifulSoup(response.text, "lxml")

        titel = self._extraheer_titel(soup)
        artikelen = self._extraheer_artikelen(soup, bwbr_id, final_url)

        if not artikelen:
            logger.warning(f"Geen artikelen gevonden voor {bwbr_id}")
            return None

        return {
            "titel": titel,
            "bwbr_id": bwbr_id,
            "url": final_url,
            "artikelen": artikelen,
        }

    def _extraheer_titel(self, soup: BeautifulSoup) -> str:
        """Extraheer de titel van de wet."""
        # Meerdere mogelijke locaties voor de titel
        for selector in ["h1.heading-title", "h1", ".wet-titel", "#wettekst h1"]:
            el = soup.select_one(selector)
            if el:
                return el.get_text(strip=True)
        return "Onbekende wet"

    def _extraheer_artikelen(
        self, soup: BeautifulSoup, bwbr_id: str, base_url: str
    ) -> List[Dict]:
        """
        Extraheer artikelen uit de wettekst.
        Probeert meerdere HTML-structuren te ondersteunen.
        """
        artikelen = []

        # Methode 1: Artikel-divs met data-attributen
        artikel_divs = soup.select("div[data-artikel], div.artikel, article.artikel")
        if artikel_divs:
            for div in artikel_divs:
                nummer = div.get("data-artikel") or div.get("id", "")
                tekst = div.get_text(separator=" ", strip=True)
                if tekst and len(tekst) > 20:
                    artikelen.append({
                        "nummer": nummer,
                        "tekst": tekst,
                        "bron": bwbr_id,
                    })
            if artikelen:
                return artikelen

        # Methode 2: Zoek naar "Artikel X" headers
        wettekst = soup.select_one("#wettekst, .wettekst, main, .content")
        if not wettekst:
            wettekst = soup

        tekst_blokken = wettekst.get_text(separator="\n").split("\n")
        huidig_artikel = None
        huidige_tekst = []

        for regel in tekst_blokken:
            regel = regel.strip()
            if not regel:
                continue

            # Detecteer artikel-headers
            artikel_match = re.match(
                r"^(Artikel\s+\d+[a-z]?\.?\s*(?:[A-Z][a-zA-Z\s]*)?)", regel, re.IGNORECASE
            )
            if artikel_match:
                if huidig_artikel and huidige_tekst:
                    artikelen.append({
                        "nummer": huidig_artikel,
                        "tekst": " ".join(huidige_tekst),
                        "bron": bwbr_id,
                    })
                huidig_artikel = artikel_match.group(1).strip()
                huidige_tekst = [regel]
            elif huidig_artikel:
                huidige_tekst.append(regel)

        # Voeg laatste artikel toe
        if huidig_artikel and huidige_tekst:
            artikelen.append({
                "nummer": huidig_artikel,
                "tekst": " ".join(huidige_tekst),
                "bron": bwbr_id,
            })

        # Methode 3: Fallback — hele tekst als één chunk per 1000 tekens
        if not artikelen:
            logger.info(f"Gebruik fallback chunking voor {bwbr_id}")
            volledige_tekst = wettekst.get_text(separator=" ", strip=True)
            # Verwijder overbodige witruimte
            volledige_tekst = re.sub(r"\s+", " ", volledige_tekst)
            chunk_grootte = 1000
            for i, start in enumerate(range(0, len(volledige_tekst), chunk_grootte)):
                chunk = volledige_tekst[start : start + chunk_grootte]
                if chunk.strip():
                    artikelen.append({
                        "nummer": f"Deel {i + 1}",
                        "tekst": chunk,
                        "bron": bwbr_id,
                    })

        return artikelen

    def haal_meerdere_wetten(
        self, bwbr_ids: List[str], voortgang_callback=None
    ) -> List[Dict]:
        """
        Haalt meerdere wetten op via hun BWBR IDs.
        Optioneel: voortgang_callback(huidige, totaal, titel) voor progressie.
        """
        resultaten = []
        totaal = len(bwbr_ids)

        for i, bwbr_id in enumerate(bwbr_ids):
            if voortgang_callback:
                voortgang_callback(i, totaal, bwbr_id)

            wet = self.haal_wet_op_bwbr(bwbr_id)
            if wet:
                resultaten.append(wet)
                logger.info(f"Opgehaald: {wet['titel']} ({len(wet['artikelen'])} artikelen)")
            else:
                logger.warning(f"Overgeslagen: {bwbr_id}")

        return resultaten


def plat_maken(wetten: List[Dict]) -> List[Dict]:
    """
    Maakt van een lijst wetten een platte lijst van chunks,
    elk met 'tekst', 'titel', 'artikel', 'bwbr_id', 'url'.
    """
    chunks = []
    for wet in wetten:
        for artikel in wet.get("artikelen", []):
            chunks.append({
                "tekst": artikel["tekst"],
                "titel": wet["titel"],
                "artikel": artikel["nummer"],
                "bwbr_id": wet["bwbr_id"],
                "url": wet.get("url", ""),
            })
    return chunks
