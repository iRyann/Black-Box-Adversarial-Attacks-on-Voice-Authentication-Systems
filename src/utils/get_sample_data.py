import io
import json
from os import system
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def parse_demo_configs():
    """Parse conf1.json et conf2.json pour extraire IDs ADV + bonafide associés."""

    # Chargement configs
    with open("data/demo_sample/conf1.json") as f:
        conf1 = json.load(f)

    with open("data/demo_sample/conf2.json") as f:
        conf2 = json.load(f)

    # Extraction tous les samples
    all_samples = []
    adv_trial_paths = []

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
        if condition == "ADV":
            trial_adv_path = f"data/demo_sample/wavs1/{trial_id}.wav"
            adv_trial_paths.append((trial_adv_path, file_id))

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

        if condition == "ADV":
            trial_adv_path = f"data/demo_sample/wavs2/{trial_id}.wav"
            adv_trial_paths.append((trial_adv_path, file_id))

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

    for adv_trial_path, file_id in adv_trial_paths:
        system(f"mv {adv_trial_path} data/spoof_adv/{file_id}.wav")

    return adv_ids, bonafide_ids, all_needed_ids


def download_bonafide(adv_ids):
    """
    Télécharge un bonafide par speaker présent dans la liste d'ADV.
    On matche par speaker_id (et pas par audio_file_name),
    et on décode les FLAC à partir des bytes du dataset HuggingFace.
    """
    df = load_dataset("Bisher/ASVspoof_2019_LA", split="test").to_pandas()

    # key == 0 <=> bonafide dans ce dataset HF
    df_bonafide = df[df["key"] == 0].copy()

    # Récupérer les lignes ADV pour connaître les speakers concernés
    adv_rows = df[df["audio_file_name"].isin(adv_ids)]
    speakers = adv_rows["speaker_id"].unique().tolist()

    Path("data/bonafide").mkdir(parents=True, exist_ok=True)

    downloaded = 0
    missing = []
    file_speaker_pairs = {}

    for spk in tqdm(speakers, desc="Downloading bonafide by speaker"):
        # Tous les bonafide de ce speaker
        spk_samples = df_bonafide[df_bonafide["speaker_id"] == spk]

        if len(spk_samples) == 0:
            print(f"Aucun bonafide trouvé pour le speaker {spk}")
            missing.append(spk)
            continue

        spk_samples = spk_samples.copy()
        spk_samples["audio_len"] = spk_samples["audio"].apply(lambda a: len(a["bytes"]))
        spk_samples = spk_samples.sort_values("audio_len", ascending=False)

        line = spk_samples.iloc[0]
        file_speaker_pairs[line["audio_file_name"]] = line["speaker_id"]
        audio_dict = line["audio"]  # {'bytes': ..., 'path': ...}
        audio_bytes = audio_dict["bytes"]

        # Décodage FLAC à partir des bytes
        data, sr = sf.read(io.BytesIO(audio_bytes))

        # Normalisation / cast optionnel
        data = np.asarray(data, dtype=np.float32)

        output_path = Path(f"data/bonafide/{spk}.wav")
        sf.write(output_path, data, sr)

        downloaded += 1

    if missing:
        pd.DataFrame({"speaker_id": missing}).to_csv(
            "data/metadata/bonafide_missing.csv", index=False
        )

    pd.DataFrame(file_speaker_pairs).to_csv(
        "data/metadata/file_speaker_pairs.csv", index=False
    )


if __name__ == "__main__":
    # adv_ids, bonafide_ids, all_needed_ids = parse_demo_configs()
    adv_ids = pd.read_csv("data/metadata/adv_ids.csv")
    download_bonafide(adv_ids)
