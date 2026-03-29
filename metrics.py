import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def print_report(df: pd.DataFrame, model_col: str) -> None:
    valid = df[df[model_col] != "erreur"].copy()
    erreurs = len(df) - len(valid)

    y_true = valid["statut_externe"]
    y_pred = valid[model_col]

    print(f"\n{'='*60}")
    print(f"Modèle : {model_col}")
    print(f"  Total commentaires : {len(df)}")
    print(f"  Erreurs de parsing : {erreurs} ({erreurs/len(df)*100:.1f}%)")
    print(f"  Commentaires évalués : {len(valid)}")
    print(f"\n--- Rapport de classification ---")
    print(classification_report(y_true, y_pred, target_names=["accepté", "refusé"], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=["accepté", "refusé"])
    print("--- Matrice de confusion ---")
    print(f"{'':15} {'prédit accepté':>15} {'prédit refusé':>14}")
    print(f"{'réel accepté':15} {cm[0][0]:>15} {cm[0][1]:>14}")
    print(f"{'réel refusé':15} {cm[1][0]:>15} {cm[1][1]:>14}")
    print(f"{'='*60}\n")
