from utils import getSpeakerID, getSampleIndex
from aasist3 import aasist3
from tqdm import tqdm
import torchaudio
import torch
import os

# Load the model from Hugging Face Hub
model = aasist3.from_pretrained("MTUCI/AASIST3")
model.eval()

def isSpoof(sample_name: str)->tuple:
    # Load and preprocess audio
    audio, sr = torchaudio.load(sample_name)
    # Ensure audio is 16kHz and mono
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Prepare input (model expects ~4 seconds of audio at 16kHz)
    # Pad or truncate to 64600 samples
    if audio.shape[1] < 64600:
        audio = torch.nn.functional.pad(audio, (0, 64600 - audio.shape[1]))
    else:
        audio = audio[:, :64600]

    # Run inference
    with torch.no_grad():
        output = model(audio)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        prediction = 'Bonafide' if prediction == 0 else 'Spoof'

        # prediction: 0 = bonafide, 1 = spoof

        # print(f"Prediction: {'Bonafide' if prediction == 0 else 'Spoof'}")
        # print(f"Confidence: {probabilities.max().item():.3f}")
        return probabilities.max().item(),prediction  

def main():
    bonafide_files = os.listdir("./data/bonafide")
    spoof_adv_files = os.listdir("./data/spoof_adv")
    spoof_sota_files = os.listdir("./data/spoof_sota")

    results_adv = []
    results_sota = []

    for spoof_adv in tqdm(spoof_adv_files):
        score, prediction = isSpoof(f"./data/spoof_adv/{spoof_adv}")
        results_adv.append((getSpeakerID(spoof_adv),getSampleIndex(spoof_adv), score, prediction))

    for spoof_sota in tqdm(spoof_sota_files):
        score, prediction = isSpoof(f"./data/spoof_sota/{spoof_sota}")
        results_sota.append((getSpeakerID(spoof_sota),getSampleIndex(spoof_sota), score, prediction))

    results_adv_string = ""
    for speaker_id,sample_index,score,prediction in results_adv:
        results_adv_string += f"{speaker_id},{sample_index},{score:.3f},{prediction}\n"

    results_sota_string = ""
    for speaker_id,sample_index,score,prediction in results_sota:
        results_sota_string += f"{speaker_id},{sample_index},{score:.3f},{prediction}\n"

    with open("./data/results/results_aasist3_adv.csv","w") as f:
        f.write("speaker_id,sample_index,confidence,prediction\n" + results_adv_string)

    with open("./data/results/results_aasist3_sota.csv","w") as f:
        f.write("speaker_id,sample_index,confidence,prediction\n" + results_sota_string)

if __name__ == "__main__":
    main()