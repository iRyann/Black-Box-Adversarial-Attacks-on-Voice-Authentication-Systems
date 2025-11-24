# Black-Box Adversarial Attacks on Voice Authentication

> Analyser *Breaking Security-Critical Voice Authentication* (Kassis & Hengartner, 2023) et évaluer si les transformations audio F1/F4/F6 conservent leur efficacité face aux spoofs générés par TTS/VC modernes (2025).

## Contexte & Question de recherche

### Problématique
- 2023 : L'article démontre qu'on peut tromper ASV+CM via 7 transformations audio (F1-F7)
- 2025 : Les TTS/VC ont progressé (naturalité accrue) : **les heuristiques F sont-elles obsolètes ?**

### Hypothèse
> Les spoofs modernes (Index TTS, 2025) sont suffisamment naturels pour que F1/F4/F6 n'apportent aucun gain en attaque.

### Approche
Comparaison expérimentale minimaliste :
- S_baseline : 12 audios bonafide (VoxCeleb)
- S_spoof : 12 spoofs
- S_spoof+F : S_spoof après application F1/F4/F6
- Métriques : Similarité ASV (cosine) + Détectabilité CM (scores)

## Analyse critique de l'article

### Contributions principales
1. Attaque black-box universelle : 6 requêtes suffisent, temps réel, transférable
2. Transformations simples mais efficaces : F1-F7 (silences, échos, filtrage, bruit)
3. Validation end-to-end : ASV+CM+STT, dont Amazon Connect (production)
4. Robustesse : Survit à la téléphonie (8 kHz, codec GSM)

### Limites identifiées
- Dataset historique : ASVspoof2019 (spoofs "basiques" vs. SOTA 2025)
- F7 coûteuse : Nécessite un ASV "shadow" (accès partiel)
- Défenses non testées : Adversarial training, détection multimodale
- Éthique : Pas de guidelines sur l'usage responsable

### Positionnement dans l'état de l'art

- Génération vocale (2019 - 2025) :
   - 2019 : Tacotron 2, WaveNet (détectables via artefacts spectraux)
   - 2023 : VALL-E, YourTTS (zero-shot, prosodie naturelle)
   - 2025 : Coqui XTTS v2, ElevenLabs v3 (indistinguables humains sur benchmarks)

-Contre-mesures (CM) :
   - 2019 : GMM-based, LFCC (vulnérables aux transformations)
   - 2023 : AASIST, Wav2Vec 2.0 SSL (robustes au bruit)
   - 2025 : Transformers multimodaux (audio + texte + contexte)

- Défenses émergentes (post-2023) :
   - Watermarking neural (AudioSeal, WavMark)
   - Détection zero-shot (CLAP embeddings)
   - Certification probabiliste (randomized smoothing)

## Protocole expérimental

### Données
- Bonafide : 12 fichiers VoxCeleb (4 speakers × 3 utterances, 3-5s/utterance)
   - Critères : Mono 16 kHz, anglais, qualité studio, speakers distincts (2M/2F)

- Spoofs : Générés via Coqui TTS (XTTS v2, zero-shot), IndexTTS, ...
   - Input : Texte des 12 utterances bonafide + 3s d'audio référence/speaker
   - Output : 12 fichiers `.wav` (même format que bonafide)

### Évaluation

- ASV - Similarité locuteur** :
   - Modèle : ECAPA (pré-entraîné VoxCeleb)
   - Métrique : Cosine similarity (bonafide vs. spoof)
   - Seuil d'acceptation : Médiane des cosines intra-speaker bonafide

- CM - Détection spoof :
   - Modèle : AASIST
   - Métrique : Score CM (bonafide=1, spoof=0) + distribution

## Limitations du MVP
- **Échantillon réduit** : 12 utterances (tendance, pas généralisation)
- **1 TTS** : SOTA mais pas ElevenLabs/Azure
- **1 CM** : AASIST
- **Pas de STT** : Intelligibilité non mesurée
- **Pas de téléphonie** : Simulation 8 kHz non testée
- 
## Références
1. Kassis, A., Hengartner, U. **Breaking Security-Critical Voice Authentication.** *IEEE Symposium on Security and Privacy (S&P)*, 2023. 
2. Repo officiel : `github.com/andrekassis/Breaking-Security-Critical-Voice-Authentication`
