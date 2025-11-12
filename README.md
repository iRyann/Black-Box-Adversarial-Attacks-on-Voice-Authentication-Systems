# Black-Box Adversarial Attacks on Voice Authentication Systems

## Résumé

-> Résumé du travail académique mêné, consistant en l'étude et l'exploitation d'un article de recherche dans le champs de l'IA appliquée à la Cybersécurité.
Objectifs :

- Comprendre, et apprécier la dynamique globale de l'article.
- Se saisir des modèles, de leur intention d'usage, de leur pertinence, et évaluer les résultats présenter au regard des métriques choisies par les auteurs.
- Expliquer les modèles utilisés, leur fonctionnement (de façon succinte), et les enjeux connexes.
- Apprécier l'article dans le contexte récent, et relativement à l'évolution de l'état de l'art.
- Confronter l'état de l'art avec le contenu (ie l'attaque) de l'article (où inversemment, c'est la même chose).
- Conclure et ouvrir sur les perspectives du domaine.

## Introduction

### Contexte de la thématique

La sécurité des systèmes : un enjeu critique, globalisé, et à toutes les échelles => À cet effet : des protocoles de sécurité tels que l'authentification vocale, utilisée à auteur de <parts de marché des acteurs majeurs> et sur lesquels reposent des systèmes d'informations de tous les secteurs (eg bancaire,<à détailler>) -- le tout, représentant des milliards (valorisations, fonds, etc.).
Problématique : Développement rapide de l'IA, avec des progrès notables en matière de production de _faux_ ; et subséquemment, l'avènement d'une période aux risques d'usurpation extrêmes.

### Sujet d'étude

-> L'article, ses auteurs et leur réalisation : Breaking Security-Critical Voice Authentication.
Une attaque agnostique, boite noire, et générique, des systèmes d'authentification vocale en condition de sécurité critique, parachevée par exploitation sur canal téléphoniques.

### Éléments de propédeutique

Préciser les termes clés, les concepts indispensables pour apprécier la thématique, et son approche ; ainsi que les éléments qui seront mentionnés au cours de la présentation.

## Revue analytique de l'article

Objectif : témoigner d'une lecture critique de ce dernier.

### Cible

VA, en particulier CMs passifs, la faillibilité des ASV étant établie.

### Contributions principales

Citation de l'article :

> Our attack can circumvent any state-of-the-art VA
> platform implementing the authentication protocol
> in its strictest form and is fully black-box, query-
> efficient, real-time, model-agnostic (transferable), and
> targeted.
> Our attack is effective against the full VA stack.
> Robustness to various defenses

Code source des auteurs disponible [ici](https://github.com/andrekassis/Breaking-Security-Critical-Voice-Authentication)

1. **Nouvelle attaque universelle**
   - Fonctionne en black-box (aucun accès interne au modèle).
   - **Temps réel**, **query-efficient** (≤ 6 essais suffisent).
   - Cible les **CMs**, maillon faible des VA modernes.
2. **Transformations audio simples mais efficaces**
   - Ajout de silences “naturels” et d’échos.
   - Amplification de certaines fréquences.
   - Suppression de bruits caractéristiques des voix synthétiques.  
      → Résultat : les signaux trompent les détecteurs sans dégrader l’empreinte vocale ni le texte.
3. **Expérimentation large**
   - Testé contre 14 contre-mesures (CMs), 5 ASVs (dont Amazon Voice ID), STTs commerciaux (Google & Azure).
   - Succès jusqu’à 99 % avec seulement 6 tentatives.
   - Première démonstration d’attaque ciblée par téléphone (over-telephony)
4. **Validation humaine** : Même des juges humains n’arrivent pas à distinguer les enregistrements falsifiés.
5. **Impact sécurité** : Met en doute la fiabilité des systèmes de biométrie vocale, largement déployés dans les services financiers.

### Hypothèses

- H1: There exist identifiable, removable, and universal key
  differences between human and machine speech.
- H2: These nuances are among the primary telltales on which
  CMs rely to make their decision

### Protocole

1. **Jeu de données audio**
   - ASVspoof2019,
2. **Génération de voix synthétique (Spoofing)**
   - Modèle de **speech synthesis (SS)** ou **voice conversion (VC)**.
3. Transformations audio (les F1–F7 de l’article)\*\*
   - Silences naturels, ajout d’échos, filtrage fréquentiel, réduction de bruit, etc.
   - Librairies:
     - Librosa,
     - SciPy,
     - PyTorch Audio.
   - Sept transformations principales (F1–F7) :
     - F1/F2 = gestion des silences.
     - F3/F5 = amplification fréquentielle.
     - F4 = ajout d’écho local.
     - F6 = réduction/remplacement du bruit.
     - F7 = régularisation via un modèle ASV “shadow”.

- Ensemble de ces modifs = **adversarial spoofing audio** crédible.
- **Tests / validation**
  - ASVspoof challenge.

### Résultats

- Attaque réussie contre l’ASV Amazon Connect Voice ID.
- WAV2VEC (CM récent, robuste au bruit téléphonique) passe de 1,8 % (baseline) à 11,6 % de succès sous attaque.
- 99 % de réussite sur d’autres combinaisons ASV+CM+STT.
  D'autres résultats sont exploitables, notamment les taux de réussites selon les technos et au gré des couches (F1, F2, ...) appliquées.

### Conclusion et ouverture

Les systèmes manquent manifestement de robustesse, Amazon a pris des mesures (CM) suite à l'article.

## Exploitation

### Points d'ancrages

Il s'agit d'un article de 2021, utilisant des audios spoof de 2019. Depuis, l'état de l'art en matière de génération IA a considérablement évolué, et on se doute que les systèmes d'authentification vocale aussi.
Sur la base de constat simple, il semble intéressant, moins de reproduire la pipeline de l'article, que de confronter cette dernière aux derniers VAS exploitables (open-sources ? gratuit ?), ou à défaut de possibilité, évaluer la pertinence de la pipeline (ie des transformations F) au vu de l'_humanisation_ des audios qui découle des progrès réalisés en matière d'IA générative.

### Évolution de l'état de l'art

- Sur la base des techniques utilisés lors de la générations des audios spoof du dataset ASVSpoofing 2019, quelle est l'évolution de l'état de l'art quant à ces dernières, et que se fait-il en matières de Speech Synthesis et de Voice Cloning ?
- Quels sont les systèmes d'authentification vocale en condition de sécurité critique à l'état de l'art, et quelles ont été leur évolution depuis 2021 ?

### Protocole

Il semble intéressant de considérer des audios utilisés par les auteurs de l'article, afin de générer des spoof sur la base des bonified avec les modèles de génération vocale à l'état de l'art ; et ce, afin de dénoter la pertinence des transformations (et à fortiori les taux d'acceptabilité des VAS).
Pour ce faire :

- On considère un sous-ensemble du sous-ensemble déjà considéré par les auteurs.
- On exploite les enregistrements réels afin de générer des spoof avec un, ou deux, modèles à l'état de l'art.
- On confronte les audios générés à quelques ASV, VAS (ie tout le systèmes, de l'ASV au STT) ?
- On dénote les résultats en appliquant les transformations F, ou non.

### Expérimentation

Il faut se renseigner sur Google Cloud ou Azure pour le STT, et concernant un usage gratuit de ces derniers pour notre expérimentation.

### Résultats

À voir !

## Conclusion et ouverture

À voir plus tard !
