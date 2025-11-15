import json
import pandas as pd
from pathlib import Path

def parse_demo_configs():
    """Parse conf1.json et conf2.json pour extraire tous les IDs uniques."""
    
    # Chargement configs
    with open("data/demo_sample/conf1.json") as f:
        conf1 = json.load(f)
    
    with open("data/demo_sample/conf2.json") as f:
        conf2 = json.load(f)
    
    # Extraction tous les samples
    all_samples = []
    
    # Conf1 : format {T1: "LA_E_8741617-ADV", ...}
    for trial_id, sample_id in conf1.items():
        parts = sample_id.rsplit('-', 1)
        file_id = parts[0]
        condition = parts[1] if len(parts) > 1 else "unknown"
        
        all_samples.append({
            'file_id': file_id,
            'condition': condition,
            'config': 'conf1',
            'trial_id': trial_id
        })
    
    # Conf2 : format {T1A: "LA_E_4468511-bonafide", T1B: "LA_E_8231152-bonafide", ...}
    for trial_id, sample_id in conf2.items():
        parts = sample_id.rsplit('-', 1)
        file_id = parts[0]
        condition = parts[1] if len(parts) > 1 else "unknown"
        
        all_samples.append({
            'file_id': file_id,
            'condition': condition,
            'config': 'conf2',
            'trial_id': trial_id
        })
    
    df = pd.DataFrame(all_samples)
    by_condition = df.groupby('condition')['file_id'].unique()
    
    bonafide_ids = by_condition.get('bonafide', [])
    spoofed_ids = by_condition.get('spoofed', [])
    adv_ids = by_condition.get('ADV', [])
    control_ids = by_condition.get('control', [])
    
    print("\n INFO - Unique IDs:\n")
    print(f"Bonafide: {len(bonafide_ids)} unique IDs")
    print(f"Spoofed: {len(spoofed_ids)} unique IDs")
    print(f"ADV: {len(adv_ids)} unique IDs")
    print(f"Control: {len(control_ids)} unique IDs")
    
    # Sauvegarde
    Path("data/metadata").mkdir(parents=True, exist_ok=True)
    
    # CSV complet
    df.to_csv("data/metadata/all_samples.csv", index=False)
    
    # CSV par condition (pour faciliter download)
    pd.DataFrame({'file_id': bonafide_ids}).to_csv(
        "data/metadata/bonafide_ids.csv", index=False
    )
    pd.DataFrame({'file_id': spoofed_ids}).to_csv(
        "data/metadata/spoofed_ids.csv", index=False
    )
    pd.DataFrame({'file_id': adv_ids}).to_csv(
        "data/metadata/adv_ids.csv", index=False
    )
    return bonafide_ids, spoofed_ids, adv_ids

if __name__ == "__main__":
    bonafide_ids, spoofed_ids, adv_ids = parse_demo_configs()
