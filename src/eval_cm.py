from speechbrain.inference.speaker import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)
score, prediction = verification.verify_files(
    "/content/example1.wav", "/content/example2.flac"
)

print(prediction, score)
