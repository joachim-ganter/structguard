# StructGuard v2.1
### Structural & Semantic Validation Layer for LLM Outputs

**Autor:** J. M. Ganter  
**ORCID:** [0009-0005-0499-2056](https://orcid.org/0009-0005-0499-2056)  
**Version:** 2.1  
**Lizenz:** GNU Affero General Public License v3.0 (AGPL-3.0)  
**Verwandte Publikation:** SENTRY-DVL v1.2 – DOI: [10.5281/zenodo.20490643](https://doi.org/10.5281/zenodo.20490643)

---

## Übersicht

StructGuard ist ein zweistufiger, deterministisch-semantischer Validierungs-Layer für strukturierte LLM-Outputs. Er bildet die strukturelle Eingangsschicht einer kombinierten Validierungspipeline zusammen mit SENTRY-DVL.

```
LLM Output
    ↓
Phase 1: Schema-Validierung (deterministisch)
    → Typfehler, fehlende Felder, ungültige Enum-Werte
    → Bei Fehler: Early-Exit ABGELEHNT
    ↓
Phase 2: Semantische Kohärenz (embedding-basiert)
    → Cosinus-Ähnlichkeit: Query ↔ Answer ↔ Kontext
    → FREIGEGEBEN / UNSICHER / ABGELEHNT
    ↓
Übergabe an SENTRY-DVL (nur wenn FREIGEGEBEN)
```

---

## Designprinzipien

**Safety-First:** Ein fälschlich blockierter Output ist unbequem. Ein fälschlich freigegebener Output in einem sicherheitskritischen Kontext ist gefährlich. StructGuard priorisiert 0% False Positives.

**Auditierbarkeit:** Jedes Verdict enthält einen vollständigen Audit-Trail mit Phase, Einzelscores und Fehlerbeschreibung.

**Komplementarität:** StructGuard übernimmt strukturelle und semantische Prüfung. SENTRY-DVL übernimmt inhaltliche Prüfung. Keine Überlappung, klare Verantwortungstrennung.

---

## Changelog v2.1

- **No-Context-Bug behoben:** Ohne Kontext-Embeddings war der maximale erreichbare Score 0.40 (unter dem Accept-Threshold von 0.48) → identische Vektoren wurden nie freigegeben. Fix: dynamische Gewichtung
- **bool-als-int-Bug behoben:** `True`/`False` wurden als gültige `int`-Werte akzeptiert. Fix: explizite bool-Prüfung vor Typvalidierung
- **Docstrings** und Inline-Kommentare vollständig ergänzt
- **StructVerdict** um `ist_abgelehnt()` und `ist_unsicher()` erweitert

---

## Installation

```bash
pip install numpy
```

StructGuard v2.1 hat bewusst minimale Abhängigkeiten. Embeddings werden extern erzeugt (z.B. mit `sentence-transformers`) und als `np.ndarray` übergeben.

---

## Schnellstart

```python
from structguard import StructGuard, StructGuardValidator, Schema
import numpy as np

# Schema definieren
schema = Schema({
    "text":      str,
    "kategorie": ["Dosierung", "Einheit", "Negation", "Semantik"],
    "quelle":    str,
})

# Nur Schema-Validierung
guard = StructGuard(schema=schema)
verdict = guard.pruefen({"text": "Metformin 500 mg", "kategorie": "Dosierung", "quelle": "Arzt"})
print(verdict.status)   # FREIGEGEBEN

# Mit semantischer Validierung
sem = StructGuardValidator()
guard_sem = StructGuard(schema=schema, semantic_validator=sem)

verdict = guard_sem.pruefen(
    answer={"text": "Metformin 500 mg", "kategorie": "Dosierung", "quelle": "Arzt"},
    query_emb=np.array([1.0, 0.0, 0.0]),
    answer_emb=np.array([0.95, 0.05, 0.0]),
    context_embs=[np.array([0.97, 0.03, 0.0])],
)
print(verdict.status)   # FREIGEGEBEN
print(verdict.score)    # 0.999
print(verdict.details)  # vollständiger Audit-Trail
```

---

## Klassen

### `Schema`
Deterministischer Strukturvalidator.
- Typprüfung (`str`, `int`, `float`, etc.)
- Enum-Validierung (Liste erlaubter Werte)
- Pflichtfeld-Prüfung
- Explizite bool-Abgrenzung bei int-Feldern

### `StructGuardValidator`
Embedding-basierter Kohärenzvalidator.
- Cosinus-Ähnlichkeit mit Epsilon-Schutz
- Dynamische Gewichtung (Fix v2.1)
- Konfigurierbare Thresholds

### `StructGuard`
Fassade – kombiniert Phase 1 und Phase 2.
- Early-Exit bei Strukturfehlern
- Vollständiger Audit-Trail im StructVerdict

### `StructVerdict`
Ergebnisobjekt mit `status`, `score`, `fehler`, `details`.
- `ist_freigegeben()` → bool
- `ist_abgelehnt()` → bool
- `ist_unsicher()` → bool

---

## Drei-Status-System

| Status | Bedeutung |
|---|---|
| `FREIGEGEBEN` | Strukturell und semantisch valide – Übergabe an SENTRY-DVL |
| `UNSICHER` | Semantisch grenzwertig – konservativ blockiert |
| `ABGELEHNT` | Strukturfehler oder semantisch inkohärent – Early-Exit |

---

## Benchmark

StructGuard v2.1 wurde im kombinierten Benchmark mit SENTRY-DVL gegen 100 Fälle aus den Domänen Medizin und Pharma getestet.

| Kategorie | Fälle | Korrekt | Accuracy |
|---|---|---|---|
| Struktur_Typ | 14 | 14 | 100% |
| Struktur_Feld | 14 | 14 | 100% |
| Struktur_Enum | 16 | 16 | 100% |
| **StructGuard gesamt** | **44** | **44** | **100%** |

Precision: 100% · False Positives: 0

---

## Lizenz

Dieses Projekt steht unter der **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- Nutzung, Modifikation und Weitergabe sind frei erlaubt
- Abgeleitete Werke und Netzwerkdienste müssen ebenfalls unter AGPL-3.0 veröffentlicht werden
- Kommerzielle Nutzung ist unter AGPL-3.0-Bedingungen erlaubt

**Kommerzielle Lizenzierung:** Für proprietäre Nutzung ohne AGPL-Pflichten ist eine kommerzielle Lizenz verfügbar. Kontakt über ORCID-Profil.

---

## Zitierung

```bibtex
@software{ganter2026structguard,
  author    = {Ganter, Joachim M.},
  title     = {StructGuard v2.1: Structural and Semantic Validation Layer 
               for LLM Outputs},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.1},
  license   = {AGPL-3.0},
  orcid     = {0009-0005-0499-2056}
}
```

---

## Verwandte Arbeiten

- **SENTRY-DVL v1.2** – Content-Level Validation Pipeline: [10.5281/zenodo.20490643](https://doi.org/10.5281/zenodo.20490643)
- **UTER** – Unified Theory of Emergent Representation: Zenodo
- **UTER-RFC-01/02/03** – Technical Standards: Zenodo

---

*StructGuard wurde entwickelt aus der Überzeugung, dass Auditierbarkeit – nicht Modellgröße – der Schlüssel zum verantwortungsvollen Einsatz von KI in sicherheitskritischen Domänen ist.*
