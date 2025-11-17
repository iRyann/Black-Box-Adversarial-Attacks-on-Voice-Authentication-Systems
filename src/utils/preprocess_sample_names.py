import json
import os
from pathlib import Path
from typing import Dict, List


def rename_bonafide(path: str):
    """
    Rename bonafide samples according to project convention.

    Convert:   LA_0012.wav
    Into:      BONA_12.wav

    Args:
        path: path of the bonafide samples folder
    """
    path = Path(path)
    files = list(path.iterdir())

    for file in files:
        if not file.is_file():
            continue

        # Example: LA_0012.wav
        parts = file.stem.split("_")
        if len(parts) < 2:
            print(f"  Skipping {file.name} (unexpected format)")
            continue

        spk_id = parts[-1].lstrip("0")  # LA_0012.wav -> 12
        new_name = f"BONA_{spk_id}.wav"
        new_path = path / new_name

        print(f" {file.name}  →  {new_name}")
        file.rename(new_path)


def rename_adv(path: str, descriptor_path: str):
    """
    Rename adversarial spoof samples using JSON descriptor.

    Input example:
        LA_E_3866634.wav

    Output:
        ADV_<speaker_id>_<index>.wav
        Example: ADV_123_2.wav

    JSON descriptor format (generated earlier):
    {
        "123": ["LA_E_3866634", "LA_E_4959553"],
        "456": ["LA_E_2855906"]
    }

    Args:
        path: folder containing ADV samples
        descriptor_path: JSON mapping speaker -> list of ADV file_ids
    """
    path = Path(path)

    # Load descriptor
    with open(descriptor_path) as f:
        adv_map: Dict[str, List[str]] = json.load(f)

    # Build reverse map: file_id -> (speaker, index)
    reverse_index = {}

    for spk, file_ids in adv_map.items():
        for idx, file_id in enumerate(file_ids, start=1):
            reverse_index[file_id] = (spk, idx)

    for file in path.iterdir():
        if not file.is_file():
            continue

        stem = file.stem  # ex: LA_E_3866634
        file_id = stem

        if file_id not in reverse_index:
            print(f"  Skipping {file.name} (not in descriptor)")
            continue

        spk, idx = reverse_index[file_id]
        parts = spk.split("_")
        spk_id = parts[-1].lstrip("0")  # LA_0012
        new_name = f"ADV_{spk_id}_{idx}.wav"
        new_path = path / new_name

        print(f" {file.name}  →  {new_name}")
        file.rename(new_path)


def rename_sota(path: str):
    """
    Rename SOTA-generated spoofs into project convention :

    Input example:
        LA_SOTA_12_3.wav   or   SOTA_12_3.wav  (depending on your generator)

    Output:
        SOTA_<speaker_id>_<index>.wav
        Example: SOTA_12_3.wav

    Args:
        path: path of the sota samples folder
    """
    path = Path(path)

    for file in path.iterdir():
        if not file.is_file():
            continue

        parts = file.stem.split("_")
        if len(parts) < 3:
            print(f"  Skipping {file.name} (unexpected format)")
            continue

        # Robust pattern extraction
        # Expected formats:
        #   LA_SOTA_<spk>_<idx>
        #   SOTA_<spk>_<idx>
        if parts[0] == "LA":
            _, _, spk, idx = parts
        elif parts[0] == "SOTA":
            _, spk, idx = parts
        else:
            print(f"  Skipping {file.name} (unknown pattern)")
            continue

        spk = spk.lstrip("0")
        idx = idx.lstrip("0")

        new_name = f"SOTA_{spk}_{idx}.wav"
        new_path = path / new_name

        print(f"🔄 {file.name}  →  {new_name}")
        file.rename(new_path)


if __name__ == "__main__":
    DATA_PATH = "data/"
    BONA_PATH = DATA_PATH + "bonafide/"
    ADV_PATH = DATA_PATH + "spoof_adv/"
    SOTA_PATH = DATA_PATH + "spoof_sota/"
    DESC_PATH = DATA_PATH + "metadata/adv_ids_by_speaker.json"

    print("\n--- Renaming BONAFIDE")
    rename_bonafide(BONA_PATH)

    print("\n--- Renaming ADVERSARIAL SPOOFS")
    rename_adv(ADV_PATH, DESC_PATH)

    print("\n--- Renaming SOTA SPOOFS")
    rename_sota(SOTA_PATH)
