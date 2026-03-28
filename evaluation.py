"""
Evaluation module for the Dutch legal RAG application.

Provides:
- TokenOverlapScorer: local unigram F1 scorer (no API calls)
- LLMJudge: wraps HuggingFace InferenceClient to score 4 metrics in a single call
- Evaluator: orchestrates a full evaluation run over a test suite
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Dutch stopwords excluded from token overlap scoring
_STOPWORDS = {
    "de", "het", "een", "van", "en", "is", "in", "op", "aan", "te", "dat",
    "die", "dit", "zijn", "er", "maar", "om", "hem", "ze", "ook", "als",
    "dan", "nog", "bij", "al", "wel", "niet", "naar", "met", "voor", "door",
    "of", "uit", "over", "zich", "tot", "was", "worden", "worden", "heeft",
    "hebben", "worden", "kan", "zijn", "wordt", "zal", "zou", "kunnen",
}

# Available LLM models for evaluation judging
EVAL_MODELLEN = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]

# Available embedding models
EMBEDDING_MODELLEN = {
    "paraphrase-multilingual-MiniLM-L12-v2 (standaard, ~50MB)": "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small (klein, snel)": "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-large (groter, nauwkeuriger)": "intfloat/multilingual-e5-large",
}


@dataclass
class EvaluationResult:
    """Result for a single test case."""
    id: str
    vraag: str
    verwacht_antwoord: str
    gegeven_antwoord: str
    retrieved_chunks: List  # List[Tuple[Dict, float]]
    latency_ms: float
    # Metrics — None means the judge call failed for this case
    token_overlap: Optional[float] = None
    trouw: Optional[int] = None          # faithfulness 1-5
    volledigheid: Optional[int] = None   # completeness 1-5
    helderheid: Optional[int] = None     # clarity 1-5
    relevantie: Optional[int] = None     # relevance 1-5


class TokenOverlapScorer:
    """Computes unigram F1 between prediction and reference.

    Stopwords are excluded before comparison so the score focuses on
    content words rather than function words.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    def score(self, prediction: str, reference: str) -> float:
        """Return F1 score in [0, 1]."""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        common = pred_set & ref_set
        if not common:
            return 0.0
        precision = len(common) / len(pred_set)
        recall = len(common) / len(ref_set)
        return round(2 * precision * recall / (precision + recall), 3)


class LLMJudge:
    """Uses a HuggingFace model to score four quality metrics in a single call.

    Scores are integers 1–5:
    1 = very poor, 3 = acceptable, 5 = excellent

    Rate limiting: enforces a minimum gap of 1.2 seconds between API calls.
    Retry logic: up to 3 attempts with exponential backoff on 429/503.
    """

    _last_call_time: float = 0.0
    _MIN_INTERVAL = 1.2  # seconds between API calls

    def __init__(self, token: str, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        from huggingface_hub import InferenceClient
        self.model = model
        self.client = InferenceClient(token=token)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - LLMJudge._last_call_time
        if elapsed < self._MIN_INTERVAL:
            time.sleep(self._MIN_INTERVAL - elapsed)

    def _call_with_retry(self, messages: List[Dict], max_tokens: int = 256) -> Optional[str]:
        """Call chat_completion with up to 3 retries on rate-limit errors."""
        for attempt in range(3):
            self._rate_limit()
            try:
                LLMJudge._last_call_time = time.monotonic()
                response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "503" in error_str:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited (attempt {attempt + 1}), waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Judge call failed: {e}")
                    return None
        logger.error("Max retries reached for judge call")
        return None

    def score(
        self,
        vraag: str,
        antwoord: str,
        context_chunks: List,
        verwacht_antwoord: str,
    ) -> Dict[str, Optional[int]]:
        """Score a single answer on four metrics. Returns dict with keys:
        trouw, volledigheid, helderheid, relevantie (each int 1-5 or None).
        """
        context_str = "\n---\n".join(
            chunk.get("tekst", "")[:800]
            for chunk, _ in context_chunks[:5]
        )

        prompt = (
            "Je bent een objectieve beoordelaar van juridische antwoorden. "
            "Beoordeel het gegeven antwoord op vier criteria. "
            "Parafrasering of samenvatting van de wettekst telt als trouw aan de bron. "
            "Geef alleen een lagere score als het antwoord feitelijk onjuiste of verzonnen informatie bevat.\n\n"
            f"VRAAG: {vraag}\n\n"
            f"OPGEHAALDE WETTEKSTEN:\n{context_str}\n\n"
            f"GEGEVEN ANTWOORD: {antwoord}\n\n"
            f"VERWACHT ANTWOORD: {verwacht_antwoord}\n\n"
            "Criteria (score 1-5):\n"
            "- trouw: bevat het antwoord uitsluitend informatie die aantoonbaar uit de wetteksten komt? "
            "Parafrasering is toegestaan. Score 5 als er geen verzonnen informatie is, score 1 bij veel hallucinaties.\n"
            "- volledigheid: dekt het antwoord de kernpunten van het verwachte antwoord? "
            "Score 5 als alle kernpunten aanwezig zijn, score 1 als vrijwel niets klopt.\n"
            "- helderheid: is het antwoord begrijpelijk en goed gestructureerd? "
            "Score 5 voor een helder antwoord, score 1 voor verwarrend of onleesbaar.\n"
            "- relevantie: beantwoordt het antwoord de gestelde vraag direct? "
            "Score 5 als de vraag volledig beantwoord is, score 1 als het antwoord de vraag negeert.\n\n"
            "Geef ALLEEN een JSON-object terug, geen uitleg:\n"
            '{"trouw": <1-5>, "volledigheid": <1-5>, "helderheid": <1-5>, "relevantie": <1-5>}'
        )

        raw = self._call_with_retry([{"role": "user", "content": prompt}])
        if raw is None:
            return {"trouw": None, "volledigheid": None, "helderheid": None, "relevantie": None}

        return self._parse_scores(raw)

    @staticmethod
    def _parse_scores(raw: str) -> Dict[str, Optional[int]]:
        """Parse JSON scores from model output, stripping markdown fences if present."""
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        # Extract the first {...} block
        match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
        if not match:
            logger.warning(f"Could not find JSON in judge response: {raw[:200]}")
            return {"trouw": None, "volledigheid": None, "helderheid": None, "relevantie": None}
        try:
            data = json.loads(match.group())
            result = {}
            for key in ("trouw", "volledigheid", "helderheid", "relevantie"):
                val = data.get(key)
                if isinstance(val, (int, float)) and 1 <= int(val) <= 5:
                    result[key] = int(val)
                else:
                    result[key] = None
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse error in judge response: {e} — raw: {raw[:200]}")
            return {"trouw": None, "volledigheid": None, "helderheid": None, "relevantie": None}


class Evaluator:
    """Orchestrates a full evaluation run over a test suite."""

    def __init__(self, hf_token: str, judge_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.token_scorer = TokenOverlapScorer()
        self.judge = LLMJudge(token=hf_token, model=judge_model)

    def run_evaluation(
        self,
        pipeline,
        test_cases: List[Dict],
        mode: str = "snel",
        top_k: int = 5,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[EvaluationResult]:
        """Run evaluation over test cases.

        Args:
            pipeline: RAGPipeline instance
            test_cases: list of test case dicts from test_cases.json
            mode: 'snel' (first 5 cases) or 'volledig' (all cases)
            top_k: number of chunks to retrieve per question
            progress_callback: called as (current, total, status_message)

        Returns:
            List of EvaluationResult instances
        """
        cases = test_cases[:5] if mode == "snel" else test_cases
        results: List[EvaluationResult] = []

        for i, tc in enumerate(cases):
            if progress_callback:
                progress_callback(i, len(cases), f"Vraag {i + 1}/{len(cases)}: {tc['vraag'][:60]}...")

            # Retrieve and generate
            try:
                details = pipeline.query_with_details(tc["vraag"], k=top_k)
                antwoord = details["antwoord"]
                retrieved = details.get("retrieved_chunks", [])
                latency = details.get("latency_ms", 0.0)
            except Exception as e:
                logger.error(f"Pipeline query failed for {tc['id']}: {e}")
                antwoord = f"Fout: {e}"
                retrieved = []
                latency = 0.0

            # Local token overlap
            token_score = self.token_scorer.score(antwoord, tc["verwacht_antwoord"])

            # LLM judge scores
            if self.judge and antwoord and not antwoord.startswith("Fout:"):
                scores = self.judge.score(
                    tc["vraag"], antwoord, retrieved, tc["verwacht_antwoord"]
                )
            else:
                scores = {"trouw": None, "volledigheid": None, "helderheid": None, "relevantie": None}

            results.append(EvaluationResult(
                id=tc["id"],
                vraag=tc["vraag"],
                verwacht_antwoord=tc["verwacht_antwoord"],
                gegeven_antwoord=antwoord,
                retrieved_chunks=retrieved,
                latency_ms=latency,
                token_overlap=token_score,
                trouw=scores.get("trouw"),
                volledigheid=scores.get("volledigheid"),
                helderheid=scores.get("helderheid"),
                relevantie=scores.get("relevantie"),
            ))

        if progress_callback:
            progress_callback(len(cases), len(cases), "Evaluatie voltooid.")

        return results

    @staticmethod
    def compute_summary(results: List[EvaluationResult]) -> Dict[str, Optional[float]]:
        """Compute mean score per metric, ignoring None values."""
        def mean_or_none(values):
            valid = [v for v in values if v is not None]
            return round(sum(valid) / len(valid), 3) if valid else None

        return {
            "token_overlap": mean_or_none([r.token_overlap for r in results]),
            "trouw": mean_or_none([r.trouw for r in results]),
            "volledigheid": mean_or_none([r.volledigheid for r in results]),
            "helderheid": mean_or_none([r.helderheid for r in results]),
            "relevantie": mean_or_none([r.relevantie for r in results]),
        }


def load_test_cases(path: str = "test_cases.json") -> List[Dict]:
    """Load test cases from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
