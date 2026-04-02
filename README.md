# medguard

Pipeline de modération de commentaires médicaux par LLM, avec évaluation comparative par rapport à une référence humaine.

## Objectif

Comparer plusieurs systèmes de modération (modération externe tierce, LLM via différents modèles et prompts) à une décision humaine de référence sur un corpus de commentaires issus d'un site d'information médicale destiné aux professionnels de santé.

Les métriques produites (precision, recall, F1 sur la classe *refusé*) permettent d'identifier le système le plus aligné avec la modération humaine.

## Installation

Requiert Python 3.13 et [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

Créer un fichier `.env` à la racine avec les clés API des fournisseurs LLM utilisés :

```env
MISTRAL_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Les identifiants de modèles suivent la convention [LiteLLM](https://docs.litellm.ai/docs/providers) (`provider/model-name`).

## Utilisation

### Rapport modération externe uniquement

Sans argument, génère uniquement le rapport comparant la modération externe à la référence humaine (aucun appel LLM) :

```bash
uv run python main.py
```

### Modération LLM

```bash
# Avec le modèle par défaut configuré dans main.py
uv run python main.py mistral/mistral-small-latest

# Choisir un prompt système différent
uv run python main.py gpt-4o --prompt 2

# Forcer le retraitement complet (ignore le checkpoint)
uv run python main.py mistral/mistral-small-latest --reset
```

### Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `model` | _(absent)_ | Identifiant LiteLLM du modèle à utiliser |
| `--prompt {1,2}` | `1` | Numéro du prompt système à envoyer au LLM |
| `--reset` | — | Supprime le checkpoint et retraite tout le corpus |

## Données d'entrée

Le CSV source (`test_sample_236.csv`) doit être séparé par `;`, encodé en `utf-8-sig`, et contenir les colonnes :

| Colonne | Description |
|---------|-------------|
| `Human` | Décision humaine de référence (`accepté` / `refusé`) |
| `Statut` | Décision de la modération externe tierce |
| `Corps Message` | Texte du commentaire à modérer |
| `pseudo` | Pseudonyme de l'auteur |
| `Date Traité` | Date de traitement |

## Sorties

Les résultats sont écrits dans le dossier `results/` :

| Fichier | Description |
|---------|-------------|
| `checkpoint.csv` | État cumulatif de tous les runs (reprise à chaud) |
| `rapport_{date}_{model}_p{n}.csv` | Export complet du DataFrame pour un run LLM |
| `rapport_{date}_{model}_p{n}.md` | Rapport Markdown avec métriques et synthèse comparative |
| `rapport_{date}_externe.md` | Rapport Markdown modération externe uniquement |

Le checkpoint permet de reprendre un run interrompu sans re-appeler le LLM sur les lignes déjà traitées. Chaque run LLM ajoute une colonne `statut_llm_{model_slug}_p{prompt}` au checkpoint.

## Prompts système

Deux prompts sont disponibles dans `moderator.py` :

- **Prompt 1** — critères détaillés adaptés au contexte médical (insulte ciblée, agression verbale, URL, diffamation, remise en cause des vaccins Covid).
- **Prompt 2** — critères génériques avec renforcement anti-prompt-injection (résistance aux tentatives de manipulation dans les commentaires).

## Structure

```
main.py          # Point d'entrée, chargement, orchestration, checkpoint
moderator.py     # Appels LLM async via LiteLLM, prompts système
metrics.py       # Calcul des métriques sklearn, génération des rapports
test_sample_236.csv   # Corpus de commentaires avec décisions humaines
results/         # Sorties générées (CSV, Markdown, checkpoint)
```
