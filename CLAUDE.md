# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**medguard** — Pipeline de modération de commentaires médicaux par LLM, avec comparaison aux décisions humaines de référence. Python 3.13, géré avec `uv`.

## Setup & Commands

```bash
# Installer les dépendances
uv sync

# Rapport modération externe uniquement (pas d'appel LLM)
uv run python main.py

# Modération LLM avec un modèle donné (prompt 1 par défaut)
uv run python main.py mistral/mistral-small-latest
uv run python main.py gpt-4o --prompt 2

# Forcer le retraitement complet (ignore le checkpoint)
uv run python main.py mistral/mistral-small-latest --reset
```

Les clés API LLM sont lues depuis `.env` via `python-dotenv` (LiteLLM gère le routing multi-provider).

## Architecture

Le pipeline se déroule en trois étapes : chargement/normalisation → modération LLM async → génération de rapports.

### Fichiers clés

- **`main.py`** — point d'entrée. Charge le CSV, normalise les colonnes, orchestre la modération et les rapports. Contient la logique de checkpoint (reprise à chaud).
- **`moderator.py`** — appels LLM via LiteLLM. Expose `moderate_batch()` (async, avec sémaphore de concurrence) et le dict `PROMPTS` (2 prompts système numérotés).
- **`metrics.py`** — calcul des métriques sklearn et génération des rapports Markdown + console.

### Données

- **`test_sample_236.csv`** — CSV source, séparateur `;`, encodage `utf-8-sig`.
- **`results/checkpoint.csv`** — état persistant des runs. Chaque run LLM ajoute une colonne `statut_llm_{model_slug}_p{prompt}`. Permet la reprise sans re-appeler les lignes déjà traitées.

### Colonnes importantes

| Colonne | Source | Rôle |
|---|---|---|
| `statut_human` | CSV colonne `Human` | **Référence** (`REF_COL`) — jamais modifiée |
| `statut_externe` | CSV colonne `Statut` | Modération externe (système tiers) |
| `statut_llm_*` | Généré par le pipeline | Décision LLM par modèle+prompt |

### Nommage des colonnes LLM

`statut_llm_{model_slug}_p{prompt}` où `model_slug` = identifiant LiteLLM avec `/` et `-` remplacés par `_`.

Exemple : `mistral/mistral-small-latest` + prompt 1 → `statut_llm_mistral_mistral_small_latest_p1`.

### Ajout d'un prompt

Ajouter une entrée dans le dict `PROMPTS` de `moderator.py`. La valeur de `--prompt` doit correspondre à une clé entière de ce dict.
