import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from metrics import print_report
from moderator import moderate_batch

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH = Path("commentaires.csv")
RESULTS_DIR = Path("results")
MODEL = "mistralai/mistral-small-2603"  # Changer ici pour tester un autre modèle

# Limiter à N lignes pour les tests (None = toutes les lignes)
TEST_LIMIT: int | None = 10
# ──────────────────────────────────────────────────────────────────────────────


def load_data(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig",
        usecols=["Statut", "Date Traité", "Corps Message", "pseudo"],
    )
    if limit:
        df = df.head(limit)
    df = df.reset_index(drop=True)
    return df


def normalize_statut(df: pd.DataFrame) -> pd.DataFrame:
    def parse(val: str) -> tuple[str, str | None]:
        if not isinstance(val, str):
            return "accepté", None
        lines = [line.strip() for line in val.splitlines() if line.strip()]
        if "Refusé" in lines:
            motif = next((line for line in lines if line not in ("Traité", "Refusé")), None)
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
    df.to_csv(checkpoint_path, index=True, encoding="utf-8-sig")


async def run(model: str, limit: int | None) -> None:
    model_slug = model.replace("/", "_").replace("-", "_")
    model_col = f"statut_llm_{model_slug}"
    motif_col = f"motif_llm_{model_slug}"
    checkpoint_path = RESULTS_DIR / "checkpoint.csv"

    print(f"Chargement des données : {CSV_PATH}")
    df = load_data(limit)
    df = normalize_statut(df)

    already_done = load_checkpoint(checkpoint_path, model_col)
    todo_idx = [i for i in df.index if i not in already_done]

    if not todo_idx:
        print("Tous les commentaires sont déjà traités (checkpoint existant).")
    else:
        print(f"Modération de {len(todo_idx)} commentaires avec {model}...")
        texts = df.loc[todo_idx, "Corps Message"].fillna("").tolist()
        results = await moderate_batch(texts, model)

        if model_col not in df.columns:
            df[model_col] = None
            df[motif_col] = None

        for idx, result in zip(todo_idx, results):
            df.at[idx, model_col] = result.get("decision", "erreur")
            df.at[idx, motif_col] = result.get("motif")

        save_checkpoint(df, checkpoint_path)
        print(f"Checkpoint sauvegardé : {checkpoint_path}")

    # Charger les colonnes checkpoint si traitement partiel précédent
    if model_col not in df.columns and checkpoint_path.exists():
        cp = pd.read_csv(checkpoint_path, encoding="utf-8-sig", index_col=0)
        if model_col in cp.columns:
            df[model_col] = cp[model_col]
            df[motif_col] = cp.get(motif_col)

    print_report(df, model_col)

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = RESULTS_DIR / f"rapport_{date_str}_{model_slug}.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Rapport CSV sauvegardé : {output_path}")


def main() -> None:
    args = sys.argv[1:]
    reset = "--reset" in args
    args = [a for a in args if a != "--reset"]
    model = args[0] if args else MODEL

    if reset:
        checkpoint_path = RESULTS_DIR / "checkpoint.csv"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Checkpoint supprimé : {checkpoint_path}")

    asyncio.run(run(model, TEST_LIMIT))


if __name__ == "__main__":
    main()
