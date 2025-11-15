import json
from pathlib import Path

import pandas as pd


def parse_demo_configs():
    """Parse conf1.json et conf2.json pour extraire IDs ADV + bonafide associés."""

    # Chargement configs
    with open("data/demo_sample/conf1.json") as f:
        conf1 = json.load(f)

    with open("data/demo_sample/conf2.json") as f:
        conf2 = json.load(f)

    # Extraction tous les samples
    all_samples = []

    # Conf1 : format {T1: "LA_E_8741617-ADV", ...}
    for trial_id, sample_id in conf1.items():
        parts = sample_id.rsplit("-", 1)
        file_id = parts[0]
        condition = parts[1] if len(parts) > 1 else "unknown"

        all_samples.append(
            {
                "file_id": file_id,
                "condition": condition,
                "config": "conf1",
                "trial_id": trial_id,
            }
        )

    # Conf2 : format {T1A: "LA_E_4468511-bonafide", T1B: "LA_E_8231152-bonafide", ...}
    for trial_id, sample_id in conf2.items():
        parts = sample_id.rsplit("-", 1)
        file_id = parts[0]
        condition = parts[1] if len(parts) > 1 else "unknown"

        all_samples.append(
            {
                "file_id": file_id,
                "condition": condition,
                "config": "conf2",
                "trial_id": trial_id,
            }
        )

    df = pd.DataFrame(all_samples)
    adv_ids = df[df["condition"] == "ADV"]["file_id"].unique()
    bonafide_ids = df[(df["condition"] == "bonafide") & (df["file_id"].isin(adv_ids))][
        "file_id"
    ].unique()

    all_needed_ids = set(adv_ids) | set(bonafide_ids)  # Union

    # Sauvegarde
    Path("data/metadata").mkdir(parents=True, exist_ok=True)

    # CSV par condition
    pd.DataFrame({"file_id": sorted(adv_ids)}).to_csv(
        "data/metadata/adv_ids.csv", index=False
    )
    pd.DataFrame({"file_id": sorted(bonafide_ids)}).to_csv(
        "data/metadata/bonafide_ids.csv", index=False
    )
    pd.DataFrame({"file_id": sorted(all_needed_ids)}).to_csv(
        "data/metadata/all_needed_ids.csv", index=False
    )

    return adv_ids, bonafide_ids, all_needed_ids


if __name__ == "__main__":
    adv_ids, bonafide_ids, all_needed_ids = parse_demo_configs()
