from speechbrain.inference.speaker import SpeakerRecognition
from utils import getSpeakerID, getSampleIndex
from tqdm import tqdm
import os

def main():
    recognitionModel = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        # run_opts={"device":"cuda"}
    )

    bonafide_files = os.listdir("./data/bonafide")
    spoof_adv_files = os.listdir("./data/spoof_adv")
    spoof_sota_files = os.listdir("./data/spoof_sota")

    results_adv = []
    results_sota = []

    for bonafide in tqdm(bonafide_files):
        bonafide_id = getSpeakerID(bonafide)

        for spoof_adv in spoof_adv_files:
            if getSpeakerID(spoof_adv) == bonafide_id:
                score, prediction = recognitionModel.verify_files(
                    f"./data/bonafide/{bonafide}", f"./data/spoof_adv/{spoof_adv}"
                )
                results_adv.append((bonafide_id,getSampleIndex(spoof_adv), score.item(), prediction.item()))

        for spoof_sota in spoof_sota_files:
            if getSpeakerID(spoof_sota) == bonafide_id:
                score, prediction = recognitionModel.verify_files(
                    f"./data/bonafide/{bonafide}", f"./data/spoof_sota/{spoof_sota}"
                )
                results_sota.append((bonafide_id, getSampleIndex(spoof_sota),score.item(), prediction.item()))

    results_adv_string = ""
    for speaker_id,sample_index,score,prediction in results_adv:
        results_adv_string += f"{speaker_id},{sample_index},{score:.3f},{prediction}\n"

    results_sota_string = ""
    for speaker_id,sample_index,score,prediction in results_sota:
        results_sota_string += f"{speaker_id},{sample_index},{score:.3f},{prediction}\n"

    with open("./data/results/results_asv_adv.csv","w") as f:
        f.write("speaker_id,sample_index,score,prediction\n" + results_adv_string)

    with open("./data/results/results_asv_sota.csv","w") as f:
        f.write("speaker_id,sample_index,score,prediction\n" + results_sota_string)

if __name__ == "__main__":
    main()
