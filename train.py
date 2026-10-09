import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

from pathlib import Path

CSV_PATH = Path("data") / "csv" / "raw" / "seattleWeather_1948-2017.csv"
GRAPHS_DIR = Path("data") / "graphs"
K = 5  # impair pour éviter les égalités au vote

df = pd.read_csv(CSV_PATH)
# print(df.head())

# clean et formate le df
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
y = df['RAIN'].astype(int)

# Suppression de la feature date et t/f pluie car c'est notre label
df.drop(['RAIN', 'DATE'], axis=1, inplace=True)

# Split l'entrainement (75%) et le test (25%)
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.25)

# Conversion en tableaux NumPy pour le calcul des distances
x_train = x_train.to_numpy(dtype=float)
x_test = x_test.to_numpy(dtype=float)
train_label = y_train.to_numpy()
test_label = y_test.to_numpy()


def KNN(x, y, k):
    # Calcule la distance euclidienne
    dist_ind = np.sqrt(np.sum((x - y) ** 2, axis=1))
    # Concat le label avec la distance
    main_arr = np.column_stack((train_label, dist_ind))
    # Sort la distance par ordre croissant
    main = main_arr[main_arr[:, 1].argsort()]
    # Calcul la fréquence d'un label sur la base de la valeur de K
    count = Counter(main[0:k, 0])
    keys, vals = list(count.keys()), list(count.values())
    if len(vals) > 1:
        if vals[0] > vals[1]:
            return int(keys[0])
        else:
            return int(keys[1])
    else:
        return int(keys[0])


def plot_data(train_label, pred_train, test_label, pred_test):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    datasets = [
        # Rapport de classification avec la data d'entrainement
        ("Entrainement", train_label, pred_train),
        # Rapport de classification pour la data test
        ("Test", test_label, pred_test),
    ]

    for ax, (name, y_true, y_pred) in zip(axes, datasets):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=["Pas de pluie", "Pluie"],
            ax=ax, colorbar=False,
        )
        ax.set_title(f"{name} — accuracy {accuracy_score(y_true, y_pred):.2%}")

        # Sauvegarde du rapport texte
        report = classification_report(y_true, y_pred)
        print(f"=== {name} ===\n{report}")
        (GRAPHS_DIR / f"rapport_{name.lower()}.txt").write_text(report, encoding="utf-8")

    fig.suptitle(f"k-NN (k={K}) | pluie à Seattle")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "matrices_confusion.png", dpi=150)
    plt.close(fig)


# Prédictions : les voisins sont toujours cherchés dans la data d'entrainement
pred_train = [KNN(x_train, row, K) for row in x_train]
pred_test = [KNN(x_train, row, K) for row in x_test]

plot_data(train_label, pred_train, test_label, pred_test)