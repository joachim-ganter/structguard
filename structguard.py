# structguard.py
# StructGuard v2.1 – Structural & Semantic Validation Layer
# Autor: J. M. Ganter
# ORCID: 0009-0005-0499-2056
#
# Changelog v2.1:
#   - No-Context-Bug behoben: dynamische Gewichtung ohne Kontext
#   - Schema-Validator erweitert (Typprüfung, Enum, Pflichtfelder)
#   - StructVerdict Objekt mit vollständigem Audit-Trail
#   - Drei-Status-System: FREIGEGEBEN / UNSICHER / ABGELEHNT
#   - Fassaden-Klasse StructGuard kombiniert beide Validierungsstufen
#
# Lizenz: GNU Affero General Public License v3.0 (AGPL-3.0)
# Für kommerzielle Lizenzen: Kontakt über ORCID-Profil

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


# ─────────────────────────────────────────────────────
# VERDICT-OBJEKT
# ─────────────────────────────────────────────────────

@dataclass
class StructVerdict:
    """
    Ergebnisobjekt der StructGuard-Validierung.

    status:  FREIGEGEBEN | UNSICHER | ABGELEHNT
    score:   finaler Kohärenz-Score (0.0 – 1.0)
    fehler:  Liste der Fehlerbeschreibungen (leer bei FREIGEGEBEN)
    details: Audit-Trail mit Phase und Einzelscores
    """
    status: str
    score: Optional[float] = None
    fehler: Optional[List[str]] = None
    details: Optional[Dict[str, Any]] = None

    def ist_freigegeben(self) -> bool:
        return self.status == "FREIGEGEBEN"

    def ist_abgelehnt(self) -> bool:
        return self.status == "ABGELEHNT"

    def ist_unsicher(self) -> bool:
        return self.status == "UNSICHER"


# ─────────────────────────────────────────────────────
# SCHEMA-VALIDATOR (Phase 1 – deterministisch)
# ─────────────────────────────────────────────────────

class Schema:
    """
    Deterministischer Schema-Validator für strukturierte LLM-Outputs.

    schema_def: Dict mit Feldname → erwarteter Typ (type) oder
                Liste erlaubter Werte (list).
    Alle Keys im schema_def gelten als Pflichtfelder.

    Beispiel:
        Schema({
            "text":      str,
            "kategorie": ["Dosierung", "Einheit", "Negation"],
            "quelle":    str,
        })
    """

    def __init__(self, schema_def: Dict[str, Any]):
        self.schema_def = schema_def

    def validate(self, obj: Dict[str, Any]) -> List[str]:
        """
        Validiert obj gegen das Schema.
        Gibt eine Liste von Fehlermeldungen zurück (leer = valide).
        """
        fehler = []

        if not isinstance(obj, dict):
            return ["Antwort ist kein Dict/JSON-Objekt."]

        for feld, erwartung in self.schema_def.items():

            # Pflichtfeld prüfen
            if feld not in obj:
                fehler.append(f"Pflichtfeld fehlt: '{feld}'")
                continue

            wert = obj[feld]

            # Typprüfung (bool explizit ausschließen bei int-Erwartung)
            if isinstance(erwartung, type):
                if erwartung is int and isinstance(wert, bool):
                    fehler.append(
                        f"Typfehler bei Feld '{feld}': bool nicht als int akzeptiert"
                    )
                elif not isinstance(wert, erwartung):
                    fehler.append(
                        f"Typfehler bei Feld '{feld}': "
                        f"erwartet {erwartung.__name__}, "
                        f"erhalten {type(wert).__name__}"
                    )

            # Enum-Prüfung
            elif isinstance(erwartung, list):
                if wert not in erwartung:
                    fehler.append(
                        f"Ungültiger Enum-Wert bei Feld '{feld}': "
                        f"'{wert}' nicht in {erwartung}"
                    )

        return fehler


# ─────────────────────────────────────────────────────
# SEMANTISCHER VALIDATOR (Phase 2 – embedding-basiert)
# ─────────────────────────────────────────────────────

class StructGuardValidator:
    """
    Embedding-basierter Kohärenz-Validator.

    Prüft die semantische Ausrichtung zwischen Query, Answer und
    optionalem Kontext über Cosinus-Ähnlichkeit.

    Gewichtung:
      MIT Kontext:  score = 0.4 * query_sim + 0.6 * context_sim
      OHNE Kontext: score = query_sim
      (Fix v2.1: kein künstlicher 0.6-Malus ohne Kontext)

    Thresholds:
      score >= accept_threshold → ACCEPT
      score >= revise_threshold → REVISE
      score <  revise_threshold → REJECT
    """

    def __init__(
        self,
        accept_threshold: float = 0.48,
        revise_threshold: float = 0.40,
    ):
        self.accept_threshold = accept_threshold
        self.revise_threshold = revise_threshold

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosinus-Ähnlichkeit mit Epsilon-Schutz gegen Nullvektoren."""
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )

    def validate(
        self,
        query_emb: np.ndarray,
        answer_emb: np.ndarray,
        context_embs: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Validiert semantische Kohärenz zwischen Query, Answer und Kontext.

        Args:
            query_emb:    Embedding des Queries (np.ndarray)
            answer_emb:   Embedding der Antwort (np.ndarray)
            context_embs: Liste von Kontext-Embeddings (optional)

        Returns:
            Dict mit decision, score, query_sim, context_sim
        """
        if context_embs is None:
            context_embs = []

        # Einzelscores berechnen
        query_sim = self._cosine(query_emb, answer_emb)
        context_sims = [self._cosine(query_emb, c) for c in context_embs]
        context_sim = float(np.mean(context_sims)) if context_sims else 0.0

        # FIX v2.1: Dynamische Gewichtung
        # Ohne Kontext: nur query_sim (max. Score = 1.0 erreichbar)
        # Mit Kontext:  gewichtete Kombination
        if context_sims:
            base_score = 0.4 * query_sim + 0.6 * context_sim
        else:
            base_score = query_sim

        final_score = base_score

        # Entscheidung
        if final_score >= self.accept_threshold:
            decision = "ACCEPT"
        elif final_score >= self.revise_threshold:
            decision = "REVISE"
        else:
            decision = "REJECT"

        return {
            "decision":    decision,
            "score":       round(final_score, 3),
            "query_sim":   round(query_sim, 3),
            "context_sim": round(context_sim, 3),
        }


# ─────────────────────────────────────────────────────
# STRUCTGUARD FASSADE (kombiniert Phase 1 + Phase 2)
# ─────────────────────────────────────────────────────

class StructGuard:
    """
    StructGuard v2.1 – Zweistufige Validierungspipeline.

    Phase 1: Schema-Validierung (hart, deterministisch)
             → Typfehler, fehlende Pflichtfelder, ungültige Enum-Werte
             → Bei Fehler: sofortiger Early-Exit mit ABGELEHNT

    Phase 2: Semantische Kohärenz (weich, embedding-basiert)
             → Cosinus-Ähnlichkeit zwischen Query, Answer und Kontext
             → Ergebnis: FREIGEGEBEN / UNSICHER / ABGELEHNT

    Verwendung in der kombinierten Pipeline mit SENTRY-DVL:
        StructGuard FREIGEGEBEN → SENTRY-DVL Pipeline
        StructGuard ABGELEHNT   → Early-Exit, kein SENTRY-DVL
    """

    def __init__(
        self,
        schema: Schema,
        semantic_validator: Optional[StructGuardValidator] = None,
    ):
        self.schema = schema
        self.semantic_validator = semantic_validator

    def pruefen(
        self,
        answer: Dict[str, Any],
        query_emb: Optional[np.ndarray] = None,
        answer_emb: Optional[np.ndarray] = None,
        context_embs: Optional[List[np.ndarray]] = None,
    ) -> StructVerdict:
        """
        Hauptmethode: Prüft answer gegen Schema und optional semantisch.

        Args:
            answer:       LLM-Output als Dict
            query_emb:    Embedding des Queries (optional)
            answer_emb:   Embedding der Antwort (optional)
            context_embs: Liste von Kontext-Embeddings (optional)

        Returns:
            StructVerdict mit status, score, fehler, details
        """

        # ── Phase 1: Schema-Validierung ──────────────────
        schema_errors = self.schema.validate(answer)
        if schema_errors:
            return StructVerdict(
                status="ABGELEHNT",
                score=0.0,
                fehler=schema_errors,
                details={"phase": "schema"},
            )

        # ── Kein semantischer Validator → nur Schema ──────
        if (self.semantic_validator is None
                or query_emb is None
                or answer_emb is None):
            return StructVerdict(
                status="FREIGEGEBEN",
                score=1.0,
                fehler=[],
                details={"phase": "schema_only"},
            )

        # ── Phase 2: Semantische Kohärenz ─────────────────
        result = self.semantic_validator.validate(
            query_emb=query_emb,
            answer_emb=answer_emb,
            context_embs=context_embs,
        )

        decision = result["decision"]
        score    = result["score"]

        if decision == "ACCEPT":
            status = "FREIGEGEBEN"
        elif decision == "REVISE":
            status = "UNSICHER"
        else:
            status = "ABGELEHNT"

        return StructVerdict(
            status=status,
            score=score,
            fehler=[] if status == "FREIGEGEBEN"
                      else [f"Semantische Entscheidung: {decision}"],
            details={"phase": "semantic", **result},
        )


# ─────────────────────────────────────────────────────
# DIREKTER TESTLAUF
# ─────────────────────────────────────────────────────

if __name__ == "__main__":

    print("StructGuard v2.1 – Testlauf\n")

    # Schema definieren
    schema = Schema({
        "text":      str,
        "kategorie": ["Dosierung", "Einheit", "Negation", "Semantik"],
        "quelle":    str,
    })
    guard = StructGuard(schema=schema)

    # Test 1: Strukturfehler
    a1 = {"text": 12345, "kategorie": "Dosierung", "quelle": "Arzt"}
    v1 = guard.pruefen(a1)
    print(f"Test 1 (Typfehler):     {v1.status} | {v1.fehler}")

    # Test 2: Korrekte Struktur
    a2 = {"text": "Metformin 500 mg zweimal täglich",
          "kategorie": "Dosierung", "quelle": "Arzt"}
    v2 = guard.pruefen(a2)
    print(f"Test 2 (korrekt):       {v2.status} | Score: {v2.score}")

    # Test 3: Mit semantischem Validator
    sem = StructGuardValidator()
    guard_sem = StructGuard(schema=schema, semantic_validator=sem)

    q = np.array([1.0, 0.0, 0.0])
    a = np.array([0.95, 0.05, 0.0])
    c = [np.array([0.97, 0.03, 0.0])]

    v3 = guard_sem.pruefen(a2, query_emb=q, answer_emb=a, context_embs=c)
    print(f"Test 3 (semantisch):    {v3.status} | Score: {v3.score}")

    # Test 4: Fix v2.1 – kein Kontext
    v4 = guard_sem.pruefen(a2,
                            query_emb=np.array([1.0, 0.0, 0.0]),
                            answer_emb=np.array([1.0, 0.0, 0.0]),
                            context_embs=[])
    print(f"Test 4 (kein Kontext):  {v4.status} | Score: {v4.score} "
          f"(Fix v2.1 aktiv)")

    print("\nStructGuard v2.1 erfolgreich geladen.")
