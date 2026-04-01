from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def print_report(df: pd.DataFrame, model_col: str, ref_col: str = "statut_human") -> None:
    valid = df[(df[model_col] != "erreur") & df[ref_col].notna()].copy()
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


def _metrics_section(df: pd.DataFrame, pred_col: str, ref_col: str) -> list[str]:
    valid = df[(df[pred_col] != "erreur") & df[ref_col].notna()].copy()
    erreurs = len(df) - len(valid)
    y_true = valid[ref_col]
    y_pred = valid[pred_col]

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

    return [
        f"**Prédicteur :** {pred_col}",
        f"**Référence :** {ref_col}",
        f"**Erreurs / lignes sans référence :** {erreurs} ({erreurs/len(df)*100:.1f}%)",
        f"**Commentaires évalués :** {len(valid)}",
        "",
        "| Classe       | precision | recall | f1-score | support |",
        "|--------------|-----------|--------|----------|---------|",
        *rows,
        "",
        "|                  | Prédit accepté | Prédit refusé |",
        "|------------------|----------------|---------------|",
        f"| Réel accepté     | {cm[0][0]:>14} | {cm[0][1]:>13} |",
        f"| Réel refusé      | {cm[1][0]:>14} | {cm[1][1]:>13} |",
        "",
    ]


def _synthesis_section(df: pd.DataFrame, ref_col: str, current_col: str | None = None) -> list[str]:
    """Tableau comparatif de tous les systèmes de modération présents dans df."""
    llm_cols = sorted(c for c in df.columns if c.startswith("statut_llm_"))
    systems = [("statut_externe", "Modération externe")] + [
        (col, col.removeprefix("statut_llm_")) for col in llm_cols
    ]

    def row_metrics(pred_col: str) -> dict:
        valid = df[(df[pred_col] != "erreur") & df[ref_col].notna()].copy()
        if valid.empty:
            return {}
        y_true = valid[ref_col]
        y_pred = valid[pred_col]
        report = classification_report(
            y_true, y_pred,
            labels=["accepté", "refusé"],
            zero_division=0,
            output_dict=True,
        )
        n = len(valid)
        correct = (y_true.values == y_pred.values).sum()
        return {
            "accuracy": correct / n,
            "precision": report["refusé"]["precision"],
            "recall": report["refusé"]["recall"],
            "f1": report["refusé"]["f1-score"],
            "n": n,
        }

    header = [
        "## Synthèse comparative",
        "",
        f"Référence : `{ref_col}` — métriques sur la classe **refusé** (positif).",
        "",
        "| Système | n | Accuracy | Precision | Recall | F1 |",
        "|---------|---|----------|-----------|--------|-----|",
    ]

    rows = []
    for col, label in systems:
        if col not in df.columns:
            continue
        m = row_metrics(col)
        if not m:
            continue
        marker = " ◀" if col == current_col else ""
        rows.append(
            f"| {label}{marker} | {m['n']} | {m['accuracy']:.2f} "
            f"| {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |"
        )

    if not rows:
        return []

    return header + rows + [""]


def generate_markdown_report_externe(
    df: pd.DataFrame,
    ref_col: str = "statut_human",
    output_path: Path = Path("results/rapport_externe.md"),
) -> None:
    lines = [
        "# Rapport de modération externe",
        "",
        f"**Date :** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Commentaires total :** {len(df)}",
        "",
        f"## Modération externe (`statut_externe` vs `{ref_col}`)",
        "",
        *_metrics_section(df, "statut_externe", ref_col),
        *_synthesis_section(df, ref_col, current_col="statut_externe"),
    ]
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport Markdown sauvegardé : {output_path}")


def generate_markdown_report(
    df: pd.DataFrame,
    model_col: str,
    model: str,
    prompt: int,
    output_path: Path,
    ref_col: str = "statut_human",
    externe_col: str | None = None,
) -> None:
    lines = [
        "# Rapport de modération",
        "",
        f"**Date :** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Modèle :** {model}",
        f"**Prompt :** p{prompt}",
        f"**Commentaires total :** {len(df)}",
        "",
    ]

    if externe_col is not None:
        lines += [
            f"## Modération externe (`{externe_col}` vs `{ref_col}`)",
            "",
            *_metrics_section(df, externe_col, ref_col),
        ]

    lines += [
        f"## Modération LLM (`{model_col}` vs `{ref_col}`)",
        "",
        *_metrics_section(df, model_col, ref_col),
        *_synthesis_section(df, ref_col, current_col=model_col),
    ]

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport Markdown sauvegardé : {output_path}")
