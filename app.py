import re
from pathlib import Path

import pandas as pd
import streamlit as st

CSV_PATH = Path("data") / "csv" / "raw" / "seattleWeather_1948-2017.csv"
GRAPHS_DIR = Path("data") / "graphs"
CONFUSION_PNG = GRAPHS_DIR / "matrices_confusion.png"
REPORTS = {
    "Test": GRAPHS_DIR / "rapport_test.txt",
    "Entrainement": GRAPHS_DIR / "rapport_entrainement.txt",
}
LABELS = {"0": "Pas de pluie (0)", "1": "Pluie (1)"}


def parse_report(text):
    """Convertit le texte de classification_report en DataFrame."""
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "accuracy":
            # ligne accuracy : seulement f1-score (= accuracy) et support
            rows["accuracy"] = [None, None, float(parts[1]), int(parts[2])]
        elif re.fullmatch(r"\d+", parts[0]) or parts[0] in ("macro", "weighted"):
            name = LABELS.get(parts[0], " ".join(parts[:-4]))
            *metrics, support = parts[-4:]
            rows[name] = [*map(float, metrics), int(support)]
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["precision", "recall", "f1-score", "support"]
    )


st.set_page_config(page_title="k-NN pluie à Seattle", page_icon="🌧️", layout="wide")

st.title("🌧️ k-NN : va-t-il pleuvoir à Seattle ?")
st.markdown(
    """
Ce projet implémente l'algorithme des **k plus proches voisins (k-NN)** en NumPy
pour prédire si une journée est pluvieuse à Seattle, à partir d'un
[dataset Kaggle](https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017)
couvrant 1948 à 2017. scikit-learn sert uniquement au découpage des données et
au calcul des métriques.
"""
)

# --- Données ---
st.header("1. Les données")
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    st.markdown(
        f"""
Le fichier contient **{len(df):,} jours**. Chaque ligne décrit :
- `PRCP` : précipitations (pouces)
- `TMAX` / `TMIN` : températures max / min (°F)
- `RAIN` : le label à prédire (`TRUE` s'il a plu)

Les lignes incomplètes sont supprimées et la colonne `DATE` est ignorée.
""".replace(",", "\u202f", 1)
    )
    st.dataframe(df.head(10), width="stretch")

    col1, col2 = st.columns(2)
    col1.metric("Jours de pluie", f"{df['RAIN'].mean():.1%}")
    col2.metric("Valeurs manquantes", int(df.isna().any(axis=1).sum()))

    st.warning(
        "⚠️ `RAIN` vaut `TRUE` exactement quand `PRCP > 0`. Garder `PRCP` comme "
        "feature revient à donner la réponse au modèle (fuite de données) : "
        "les scores ci-dessous sont donc optimistes."
    )
else:
    st.error(f"Dataset introuvable : `{CSV_PATH}`")

# --- Méthode ---
st.header("2. La méthode")
st.markdown(
    """
1. Le dataset est découpé en **75 % entrainement / 25 % test**.
2. Pour chaque jour à classer, on calcule la **distance euclidienne** avec
   tous les jours d'entrainement.
3. On garde les **k = 5** plus proches et on prend le **vote majoritaire**
   (k impair pour éviter les égalités).

Les voisins sont toujours cherchés dans l'entrainement. Les features ne sont pas
normalisées : les températures (dizaines de °F) pèsent donc bien plus que
`PRCP` (souvent < 1) dans la distance.
"""
)

# --- Résultats ---
st.header("3. Résultats")
st.subheader("Matrices de confusion")
st.markdown(
    "Chaque case compte les jours selon leur classe réelle (lignes) et la classe "
    "prédite (colonnes). La diagonale correspond aux bonnes prédictions."
)
if CONFUSION_PNG.exists():
    st.image(str(CONFUSION_PNG), width="stretch")
else:
    st.info("Graphique absent : lancez `uv run train.py` pour le générer.")

st.subheader("Rapports de classification")
tabs = st.tabs(list(REPORTS))
for tab, (name, path) in zip(tabs, REPORTS.items()):
    with tab:
        if not path.exists():
            st.info("Rapport absent : lancez `uv run train.py` pour le générer.")
            continue
        report = parse_report(path.read_text(encoding="utf-8"))
        if "accuracy" in report.index:
            st.metric("Accuracy", f"{report.loc['accuracy', 'f1-score']:.0%}")
        st.dataframe(report, width="stretch")
        with st.expander("Rapport brut"):
            st.code(path.read_text(encoding="utf-8"), language="text")

st.markdown(
    """
Le score d'entrainement est mécaniquement plus élevé : chaque jour d'entrainement
se retrouve parmi ses propres voisins (distance 0). C'est le score **test** qui
mesure la capacité du modèle à généraliser.
"""
)

# --- Lexique ---
st.header("4. Lexique des métriques")
st.markdown(
    """
| Terme | Signification |
| --- | --- |
| precision | VP / (VP + FP) : part des prédictions « pluie » qui sont justes |
| recall | VP / (VP + FN) : part des vrais jours de pluie détectés |
| f1-score | 2 × (precision × recall) / (precision + recall) |
| support | nombre de jours réels de la classe dans les données évaluées |
| accuracy | part de tous les jours correctement classés |
| macro avg | moyenne simple des métriques des deux classes |
| weighted avg | moyenne des métriques des deux classes pondérée par le support |

*VP = vrais positifs, FP = faux positifs, FN = faux négatifs.*
"""
)
