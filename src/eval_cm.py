import os

from speechbrain.inference.speaker import SpeakerRecognition


def getSpeakerID(sample_name: str):
    parts = sample_name.split("_")
    return parts[1]


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
    results_spoof = []

    for bonafide in bonafide_files:
        bonafide_id = getSpeakerID(bonafide)

        for spoof_adv in spoof_adv_files:
            if getSpeakerID(spoof_adv) == bonafide_id:
                score, prediction = recognitionModel.verify_files(
                    f"./data/bonafide/{bonafide}", f"./data/spoof_adv/{spoof_adv}"
                )
                results_adv.append(bonafide_id, score, prediction)

        for spoof_sota in spoof_sota_files:
            if getSpeakerID(spoof_sota) == bonafide_id:
                score, prediction = recognitionModel.verify_files(
                    f"./data/bonafide/{bonafide}", f"./data/spoof_sota/{spoof_sota.wav}"
                )
                results_sota.append(bonafide_id, score, prediction)


if __name__ == "__main__":
    main()
