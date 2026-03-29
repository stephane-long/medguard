from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def print_report(df: pd.DataFrame, model_col: str, ref_col: str = "statut_externe") -> None:
    valid = df[df[model_col] != "erreur"].copy()
    erreurs = len(df) - len(valid)

    y_true = valid[ref_col]
    y_pred = valid[model_col]

    print(f"\n{'='*60}")
    print(f"Modèle : {model_col}")
    print(f"Référence : {ref_col}")
    print(f"  Total commentaires : {len(df)}")
    print(f"  Erreurs de parsing : {erreurs} ({erreurs/len(df)*100:.1f}%)")
    print(f"  Commentaires évalués : {len(valid)}")
    print(f"\n--- Rapport de classification ---")
    print(classification_report(y_true, y_pred, labels=["accepté", "refusé"], target_names=["accepté", "refusé"], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=["accepté", "refusé"])
    print("--- Matrice de confusion ---")
    print(f"{'':15} {'prédit accepté':>15} {'prédit refusé':>14}")
    print(f"{'réel accepté':15} {cm[0][0]:>15} {cm[0][1]:>14}")
    print(f"{'réel refusé':15} {cm[1][0]:>15} {cm[1][1]:>14}")
    print(f"{'='*60}\n")


def generate_markdown_report(
    df: pd.DataFrame,
    model_col: str,
    model: str,
    prompt: int,
    output_path: Path,
    ref_col: str = "statut_externe",
) -> None:
    valid = df[df[model_col] != "erreur"].copy()
    erreurs = len(df) - len(valid)

    y_true = valid[ref_col]
    y_pred = valid[model_col]

    report = classification_report(
        y_true, y_pred,
        labels=["accepté", "refusé"],
        target_names=["accepté", "refusé"],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=["accepté", "refusé"])

    def fmt(v: float) -> str:
        return f"{v:.2f}"

    classes = ["accepté", "refusé", "weighted avg"]
    rows = []
    for cls in classes:
        r = report.get(cls, report.get("weighted avg", {}))
        rows.append(
            f"| {cls:<12} | {fmt(r['precision']):>9} | {fmt(r['recall']):>6} "
            f"| {fmt(r['f1-score']):>8} | {int(r['support']):>7} |"
        )

    lines = [
        "# Rapport de modération",
        "",
        f"**Date :** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Modèle :** {model}",
        f"**Prompt :** p{prompt}",
        f"**Référence :** {ref_col}",
        f"**Commentaires traités :** {len(df)}",
        f"**Erreurs de parsing :** {erreurs} ({erreurs/len(df)*100:.1f}%)",
        f"**Commentaires évalués :** {len(valid)}",
        "",
        "## Métriques",
        "",
        "| Classe       | precision | recall | f1-score | support |",
        "|--------------|-----------|--------|----------|---------|",
        *rows,
        "",
        "## Matrice de confusion",
        "",
        "|                  | Prédit accepté | Prédit refusé |",
        "|------------------|----------------|---------------|",
        f"| Réel accepté     | {cm[0][0]:>14} | {cm[0][1]:>13} |",
        f"| Réel refusé      | {cm[1][0]:>14} | {cm[1][1]:>13} |",
        "",
    ]

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport Markdown sauvegardé : {output_path}")
