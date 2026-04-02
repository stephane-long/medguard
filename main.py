import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd

from metrics import (
    generate_markdown_report,
    generate_markdown_report_externe,
    print_report,
)
from moderator import PROMPTS, moderate_batch

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH = Path("test_sample_236.csv")
RESULTS_DIR = Path("results")
MODEL = "mistral/mistral-small-latest"  # Changer ici pour tester un autre modèle

# Limiter à N lignes pour les tests (None = toutes les lignes)
TEST_LIMIT: int | None = None
# ──────────────────────────────────────────────────────────────────────────────


def load_data(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        CSV_PATH,
        sep=";",
        encoding="utf-8-sig",
        usecols=["Statut", "Date Traité", "Human", "Corps Message", "pseudo"],
    )
    if limit:
        df = df.head(limit)
    df = df.reset_index(drop=True)
    return df


def normalize_human(df: pd.DataFrame) -> pd.DataFrame:
    def parse(val) -> str | None:
        if pd.isna(val):
            return None
        return str(val).strip()

    df["statut_human"] = df["Human"].apply(parse)
    return df


def normalize_statut(df: pd.DataFrame) -> pd.DataFrame:
    def parse(val: str) -> tuple[str, str | None]:
        if not isinstance(val, str):
            return "accepté", None
        tokens = val.replace("\n", " ").split()
        if "Refusé" in tokens:
            motif = next((t for t in tokens if t not in ("Traité", "Refusé")), None)
            return "refusé", motif
        return "accepté", None

    parsed = df["Statut"].apply(parse)
    df["statut_externe"] = parsed.apply(lambda x: x[0])
    df["motif_externe"] = parsed.apply(lambda x: x[1])
    return df


def load_checkpoint(checkpoint_path: Path, model_col: str) -> set[int]:
    if not checkpoint_path.exists():
        return set()
    try:
        cp = pd.read_csv(checkpoint_path, encoding="utf-8-sig")
        if model_col in cp.columns:
            return set(cp.index[cp[model_col].notna()])
    except Exception:
        pass
    return set()


def save_checkpoint(df: pd.DataFrame, checkpoint_path: Path) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    if checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path, encoding="utf-8-sig", index_col=0)
        for col in existing.columns:
            if col not in df.columns:
                df[col] = existing[col]
    if "statut_ref" not in df.columns:
        df["statut_ref"] = None
    df.to_csv(checkpoint_path, index=True, encoding="utf-8-sig")


REF_COL = "statut_human"


async def run(model: str, limit: int | None, prompt: int = 1) -> None:
    model_slug = model.replace("/", "_").replace("-", "_")
    model_col = f"statut_llm_{model_slug}_p{prompt}"
    motif_col = f"motif_llm_{model_slug}_p{prompt}"
    checkpoint_path = RESULTS_DIR / "checkpoint.csv"

    print(f"Chargement des données : {CSV_PATH}")
    df = load_data(limit)
    df = normalize_statut(df)
    df = normalize_human(df)

    already_done = load_checkpoint(checkpoint_path, model_col)
    todo_idx = [i for i in df.index if i not in already_done]

    if not todo_idx:
        print("Tous les commentaires sont déjà traités (checkpoint existant).")
        # S'assurer que statut_ref existe même sans nouveau traitement
        if checkpoint_path.exists():
            cp = pd.read_csv(checkpoint_path, encoding="utf-8-sig", index_col=0)
            if "statut_ref" not in cp.columns:
                cp["statut_ref"] = None
                cp.to_csv(checkpoint_path, index=True, encoding="utf-8-sig")
                print("Colonne statut_ref ajoutée au checkpoint.")
    else:
        print(
            f"Modération de {len(todo_idx)} commentaires avec {model} (prompt p{prompt})..."
        )
        texts = df.loc[todo_idx, "Corps Message"].fillna("").tolist()
        results = await moderate_batch(texts, model, prompt=prompt)

        if model_col not in df.columns:
            df[model_col] = None
            df[motif_col] = None

        for idx, result in zip(todo_idx, results):
            df.at[idx, model_col] = result.get("decision", "erreur")
            df.at[idx, motif_col] = result.get("motif")

        save_checkpoint(df, checkpoint_path)
        print(f"Checkpoint sauvegardé : {checkpoint_path}")

    # Charger toutes les colonnes du checkpoint dans df (runs courant + précédents)
    if checkpoint_path.exists():
        cp = pd.read_csv(checkpoint_path, encoding="utf-8-sig", index_col=0)
        for col in cp.columns:
            if col not in df.columns:
                df[col] = cp[col]

    print_report(df, "statut_externe", REF_COL)
    print_report(df, model_col, REF_COL)

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"rapport_{date_str}_{model_slug}_p{prompt}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Rapport CSV sauvegardé : {csv_path}")

    md_path = RESULTS_DIR / f"rapport_{date_str}_{model_slug}_p{prompt}.md"
    generate_markdown_report(
        df, model_col, model, prompt, md_path, REF_COL, externe_col="statut_externe"
    )


def run_externe_only(limit: int | None) -> None:
    print(f"Chargement des données : {CSV_PATH}")
    df = load_data(limit)
    df = normalize_statut(df)
    df = normalize_human(df)

    # Charger les colonnes des runs LLM précédents pour la synthèse
    checkpoint_path = RESULTS_DIR / "checkpoint.csv"
    if checkpoint_path.exists():
        cp = pd.read_csv(checkpoint_path, encoding="utf-8-sig", index_col=0)
        for col in cp.columns:
            if col not in df.columns:
                df[col] = cp[col]

    print_report(df, "statut_externe", REF_COL)

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"rapport_{date_str}_externe.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Rapport CSV sauvegardé : {csv_path}")

    md_path = RESULTS_DIR / f"rapport_{date_str}_externe.md"
    generate_markdown_report_externe(df, REF_COL, md_path)


def main() -> None:
    """
    Point d'entrée du programme.

    Paramètres de ligne de commande
    --------------------------------
    model (optionnel)
        Identifiant du modèle LLM à utiliser via LiteLLM
        (ex: "mistral/mistral-small-latest", "gpt-4.1-nano", "claude-sonnet-4-6").
        - Si fourni  : lance la modération LLM sur les commentaires, puis génère
          un rapport comparant à la fois la modération externe et la modération LLM
          à la référence humaine (colonne `statut_human`).
        - Si absent  : génère uniquement le rapport de modération externe vs humain,
          sans aucun appel à un LLM.

    --prompt {1, 2}  (défaut : 1)
        Numéro du prompt système envoyé au LLM pour guider la décision de modération.
        Chaque prompt correspond à une stratégie d'instruction différente définie
        dans moderator.py. Ignoré si aucun modèle n'est fourni.

    --reset
        Supprime le fichier de checkpoint avant d'exécuter la modération.
        Utile pour forcer un retraitement complet de tous les commentaires,
        par exemple après un changement de prompt ou de modèle.
        Sans ce flag, les commentaires déjà traités sont ignorés (reprise à chaud).
    """
    parser = argparse.ArgumentParser(
        description="Modération de commentaires médicaux — comparaison vs référence humaine."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Modèle LLM à utiliser via LiteLLM (ex: mistral/mistral-small-latest, gpt-4o, claude-sonnet-4-6). "
        "Si absent, génère uniquement le rapport de modération externe.",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=1,
        choices=sorted(PROMPTS),
        help="Numéro du prompt système à utiliser (défaut : 1).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprime le checkpoint avant de lancer la modération.",
    )
    args = parser.parse_args()

    if args.reset:
        checkpoint_path = RESULTS_DIR / "checkpoint.csv"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Checkpoint supprimé : {checkpoint_path}")

    if args.model is None:
        run_externe_only(TEST_LIMIT)
    else:
        asyncio.run(run(args.model, TEST_LIMIT, args.prompt))


if __name__ == "__main__":
    main()
