import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_json_files(directory: str) -> pd.DataFrame:
    """Charge tous les fichiers JSON d'un répertoire et les combine en un DataFrame.

    Chaque fichier JSON doit être un dictionnaire ou une liste de
    dictionnaires avec des paires clé/valeur.  Une colonne
    supplémentaire ``source_file`` est ajoutée pour indiquer la
    provenance des données.  Les clés absentes dans certains fichiers
    seront renseignées avec ``NaN``.

    Args:
        directory: Chemin du dossier contenant les fichiers JSON.

    Returns:
        DataFrame concaténé contenant l'ensemble des données.
    """
    records: List[pd.DataFrame] = []
    for filename in os.listdir(directory):
        if not filename.lower().endswith(".json"):
            continue
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data: Any = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Impossible de lire {filepath}: {exc}")
            continue

        # Normalise les données vers une liste de dictionnaires
        rows: List[Dict[str, Any]]
        if isinstance(data, dict):
            # Un seul enregistrement dans un dict
            rows = [data]
        elif isinstance(data, list):
            rows = []
            for item in data:
                if isinstance(item, dict):
                    rows.append(item)
                else:
                    rows.append({"value": item})
        else:
            # Valeur simple, la convertir en dict
            rows = [{"value": data}]
        df = pd.DataFrame(rows)
        df["source_file"] = filename
        records.append(df)

    if not records:
        raise FileNotFoundError(
            f"Aucun fichier JSON trouvé dans le dossier '{directory}'."
        )
    return pd.concat(records, ignore_index=True)


def find_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Retourne la liste des colonnes numériques du DataFrame.

    Args:
        df: DataFrame chargé.

    Returns:
        Liste des noms de colonnes ayant un type numérique.
    """
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def find_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Retourne la liste des colonnes catégorielles (types object ou booléen)."""
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            # exclure la colonne source_file
            if col != "source_file":
                categorical_cols.append(col)
    return categorical_cols


def pretty_plot(directory: str = "/results") -> None:
    """Génère un graphique esthétique à partir des fichiers JSON d'un dossier.

    Ce graphique sélectionne automatiquement la première colonne
    numérique disponible et, si une colonne catégorielle est trouvée,
    produit un boxplot de cette variable numérique en fonction de
    la catégorie.  Si aucune colonne catégorielle n'est trouvée, un
    histogramme de la variable numérique est dessiné.  L'utilisation
    de `seaborn.set_theme(style="whitegrid")` applique un thème
    « grille blanche » recommandé pour les ensembles de données avec
    beaucoup de valeurs【337353910113815†L124-L133】.  La fonction
    `sns.despine()` supprime les bordures inutiles du graphique【337353910113815†L170-L174】.

    Args:
        directory: Chemin du dossier contenant les fichiers JSON à visualiser.
    """
    # Charger les données
    df = load_json_files(directory)
    numeric_cols = find_numeric_columns(df)
    if not numeric_cols:
        raise ValueError(
            "Aucune colonne numérique n'a été trouvée dans les fichiers JSON."
        )
    num_col = numeric_cols[0]
    categorical_cols = find_categorical_columns(df)

    # Définir le style du graphique
    sns.set_theme(
        style="whitegrid"
    )  # configure un style esthétique avec grille【337353910113815†L124-L133】

    plt.figure(figsize=(10, 6))
    if categorical_cols:
        cat_col = categorical_cols[0]
        # Créer un boxplot montrant la distribution de la variable numérique selon la catégorie
        sns.boxplot(x=cat_col, y=num_col, data=df, palette="muted")
        plt.title(f"Distribution de '{num_col}' selon '{cat_col}'")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
    else:
        # Si aucune catégorie, on trace un histogramme de la variable numérique
        sns.histplot(df[num_col], kde=True, color="skyblue")
        plt.title(f"Distribution de '{num_col}'")
        plt.xlabel(num_col)
        plt.ylabel("Nombre d'observations")

    # Supprimer les spines supérieure et droite pour un rendu plus épuré
    sns.despine()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Appeler la fonction avec le répertoire par défaut
    try:
        pretty_plot("/results")
    except Exception as e:
        print(f"Erreur lors de la génération du graphique : {e}")
