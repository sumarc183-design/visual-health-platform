# Routeur de modalité (J30)

Résultat du 9 novembre 2026 : le routeur reconnaît 499 fonds d'œil sur 500 et 499
dermoscopies sur 500 sur des sources qu'il n'a pas vues pendant l'entraînement, et
n'envoie aucune des 500 photos ordinaires à un modèle spécialisé. **Un critère fixé
avant le run n'est pas atteint** : un fond d'œil a été envoyé au modèle de dermoscopie.

Plan fixé et commité avant le run (`2a451bf`) : docstring de
[`scripts/train_modality_router.py`](../scripts/train_modality_router.py). Résultats
complets : [`results/modality_router.json`](../results/modality_router.json).

## Méthode

- Embeddings DINOv2 ViT-B/14 figés (Apache 2.0), image brute (petit côté 256 px,
  recadrage central 224 px), puis régression logistique à trois classes, sans réglage.
- Probabilité maximale sous 0,9 : « inconnu », aucun modèle spécialisé.
- Ajustement : 1 000 images par classe (ODIR, ISIC 2019 entraînement, Imagenette).
  Test : 500 images par classe d'autres sources ou découpages (RFMiD, ISIC 2019 test,
  Imagenette validation).

## Résultats sur le test

| Image réelle | → fond d'œil | → dermoscopie | → inconnu | Bien routées [IC 95 %] |
|---|---|---|---|---|
| Fond d'œil (RFMiD) | 499 | **1** | 0 | 0,998 [0,989 ; 1,000] |
| Dermoscopie (ISIC test) | 0 | 499 | 1 | 0,998 [0,989 ; 1,000] |
| Photo ordinaire (Imagenette) | 0 | 0 | 500 | 1,000 [0,993 ; 1,000] |

| Critère fixé avant le run | Résultat |
|---|---|
| Rappel fond d'œil ≥ 0,95 | Atteint (0,998) |
| Rappel dermoscopie ≥ 0,95 | Atteint (0,998) |
| Aucune image envoyée au mauvais modèle spécialisé | **Non atteint** (1 sur 1 000) |
| Photos ordinaires envoyées à un modèle spécialisé ≤ 2 % | Atteint (0 %, IC [0 ; 0,7 %]) |

Sur les 300 images de contrôle par classe (mêmes sources que l'ajustement), une seule
erreur : une dermoscopie classée « inconnu ».

## Les deux erreurs du test (examinées après le run, sans rien changer)

- **Fond d'œil envoyé en dermoscopie** (RFMiD test, image 443, probabilité 0,92) : rétine
  très pathologique, jaunâtre, couverte de lésions ; sa texture ressemble à de la peau.
  C'est une vraie erreur, et le cas le plus gênant : une rétine malade est justement celle
  qu'il ne faut pas mal router.
- **« Dermoscopie » classée inconnue** (ISIC_0035068) : c'est un dessin, pas une lésion.
  Le fichier officiel lui donne `score_weight = 0` et `validation_weight = 0` : une image
  piège du challenge, jamais notée. Le routeur a raison. Sans elle, la dermoscopie est
  reconnue à 100 %.

## Conséquence pour le parcours de prédiction

Le routeur n'est pas branché automatiquement dans `predict.py`. Avec un critère non
atteint, le choix proposé est de l'utiliser comme garde-fou (refuser les images
« inconnu » et signaler une modalité inattendue), le choix de la modalité restant
déclaré par l'utilisateur. À décider avec le tuteur.

## Limites

- **Contrôle non indépendant** (audit du 9 novembre) : ajustement et contrôle ont été
  tirés image par image, si bien que 70 patients ODIR et une lésion ISIC sont communs aux
  deux. Les chiffres du contrôle sont donc optimistes ; le test (autres sources) n'est pas
  concerné. Avant le prochain run : découper d'abord par patient et par lésion, puis tirer
  les images. Le résultat actuel, critère non atteint compris, reste tel quel.
- « Inconnu » n'est représenté que par des photos d'Imagenette : photos de peau sans
  dermoscope, radiographies, captures d'écran ou documents ne sont pas testés.
- Les images ISIC viennent du début des archives (premiers fichiers), pas d'un tirage dans
  toute la base.
- Une seule erreur croisée sur 1 000 : sa fréquence réelle est mal connue (IC jusqu'à
  1,1 % pour le fond d'œil).

## Protocole 2 (10 novembre 2026) : inconnus variés, même erreur résiduelle

Plan fixé et commité avant le run (`979271c`, docstring de
[`scripts/train_modality_router_v2.py`](../scripts/train_modality_router_v2.py)), après
l'audit du 9 novembre. Résultats : [`results/modality_router_v2.json`](../results/modality_router_v2.json).

- Ajustement et contrôle séparés par patient (ODIR) et par lésion (ISIC) : aucun groupe
  commun, vérifié sur les vrais tirages.
- « Inconnu » appris sur des photos (Imagenette), des radiographies du thorax et des
  frottis sanguins (MedMNIST, CC BY 4.0).
- Test sur des sources jamais vues, dont trois types d'images médicales absents de
  l'ajustement et une seconde source de fonds d'œil.

| Test (sources jamais vues) | Images | Bien routées, protocole 2 | Tête du protocole 1 |
|---|---|---|---|
| Fond d'œil, RFMiD | 500 | 499 (1 envoyée en dermoscopie) | 499 (même image) |
| Fond d'œil, RetinaMNIST (DeepDRiD) | 300 | 300 | 300 |
| Dermoscopie, ISIC test | 500 | 500 | 500 |
| Photos (Imagenette) | 300 | 300 rejetées | 300 rejetées |
| OCT de la rétine | 300 | 300 rejetées | 300 rejetées |
| Échographie mammaire | 156 | 156 rejetées | 156 rejetées |
| Scanner (CT) | 300 | 300 rejetées | 300 rejetées |

Contrôle (groupes disjoints) : 900 images sur 900 bien routées.

**Critères** : tous atteints sauf « aucune erreur croisée ». La seule erreur est la même
image qu'au protocole 1 (RFMiD 443, rétine très pathologique), envoyée en dermoscopie avec
une confiance encore plus haute (0,955 contre 0,922). Ajouter des inconnus médicaux n'y
change rien : c'est un problème de frontière entre fond d'œil malade et peau, pas de
reconnaissance des inconnus.

**Lecture** : le refus des images hors cible tient sur trois types d'images médicales
jamais vus (0 sur 756 envoyée à un modèle spécialisé), et les deux têtes se comportent de
la même façon. Le risque restant est une rétine très atteinte prise pour de la peau. Le
routeur reste non branché automatiquement ; l'usage proposé est toujours celui de
garde-fou (refuser les inconnus), la modalité restant déclarée.

Limites : images MedMNIST déjà redimensionnées ; toujours pas de photos de peau prises au
téléphone ni de vues externes de l'œil ; une seule erreur croisée, sa fréquence réelle
reste mal connue.

## Test de résistance : fonds d'œil de mauvaise qualité (10 novembre 2026)

Plan fixé avant le calcul (`scripts/modality_router_quality_stress.py`, commit `c475422`) :
têtes des protocoles 1 et 2 inchangées, appliquées aux 800 photos de validation et
d'évaluation de DeepDRiD, séparées selon la qualité notée par ses experts.

| DeepDRiD | Images | Reconnues fond d'œil (les deux têtes) | Envoyées en dermoscopie | Probabilité de dermoscopie maximale |
|---|---|---|---|---|
| Qualité suffisante | 362 | 362 | 0 | 0,001 |
| Qualité insuffisante (floues, sombres, mal cadrées) | 438 | 438 | 0 | 0,0002 |

Critères atteints : aucune image de mauvaise qualité n'approche la dermoscopie.
L'erreur résiduelle des protocoles 1 et 2 (RFMiD 443) ne vient donc pas de la qualité
d'image : c'est une rétine très pathologique, jaunâtre et couverte de lésions, un cas
rare plutôt qu'un défaut de prise de vue. Limite : DeepDRiD vient surtout d'un site,
et sa rétinopathie diabétique ne couvre pas toutes les pathologies sévères.

## Test de résistance : fonds d'œil pathologiques de RFMiD (11 novembre 2026)

**Résultat : l'erreur restante n'est pas liée à la gravité, mais à une seule pathologie,
la hyalose astéroïde (AH).** Les deux seules rétines envoyées en dermoscopie sur 3 200
portent cette seule annotation : 2 des 25 images AH de RFMiD (8 % [1 % ; 26 %]). Aucune
autre pathologie, et aucune image à plusieurs pathologies, n'est mal routée.

Plan fixé avant le calcul (`scripts/modality_router_pathology_stress.py`, commit
`0adecc0`) : têtes des protocoles 1 et 2 inchangées, seuil 0,9, appliquées aux 3 200
photos de RFMiD (le routeur n'a jamais été ajusté sur RFMiD ; 500 images du test avaient
servi à son évaluation). Critère fixé : erreur « rare » si la borne haute de l'IC à 95 %
du taux d'envoi en dermoscopie chez les images malades est au plus 0,5 %.

| RFMiD | Images | Protocole 1 : dermoscopie / inconnu | Protocole 2 : dermoscopie / inconnu |
|---|---|---|---|
| Sans pathologie (`Disease_Risk` = 0) | 669 | 0 / 0 | 0 / 0 |
| Une pathologie | 1 782 | 2 / 2 | 2 / 1 |
| Deux pathologies | 606 | 0 / 0 | 0 / 0 |
| Trois ou plus | 143 | 0 / 0 | 0 / 0 |
| **Malades, taux en dermoscopie** | 2 531 | 0,08 % [0,01 % ; 0,29 %] | 0,08 % [0,01 % ; 0,29 %] |

- **Critère atteint** pour les deux têtes : borne haute 0,29 %, sous 0,5 %.
- **Images en cause** : `test/443` (probabilité de dermoscopie 0,92 puis 0,96) et
  `train/1547` (0,98 dans les deux têtes), toutes deux annotées AH seulement.
- **Séparation nette** : hors ces deux images, la probabilité de dermoscopie ne dépasse
  jamais 0,10.

**Lecture** : la hyalose astéroïde est une affection du vitré, souvent bénigne : de
nombreux dépôts blanc-jaune flottent devant la rétine. Leur texture ponctuée ressemble
à de la peau sous dermoscope. La description antérieure de `test/443` (« rétine très
pathologique ») est donc à corriger : c'est un aspect particulier, pas une gravité.
Conséquence : un routeur bloquant écarterait environ 1 image AH sur 12 ; en
avertissement, avec la modalité déclarée, ces images restent analysables.

Limites : une seule source (RFMiD, Inde) ; seulement 25 images AH, d'où un intervalle
large ; les images n'ont pas été revues par un ophtalmologiste ; pas de correction pour
comparaisons multiples entre pathologies (lecture descriptive). Piste, sans engagement :
ajouter des images AH d'autres sources, ou en ajouter à l'ajustement d'un protocole 3,
avec un plan fixé avant.

Traçabilité : `results/modality_router_pathology_stress.json`, code `0adecc0`,
DINOv2-B (empreinte locale `ba7954cb2db99604`), run CPU d'environ 21 minutes.

## Test de résistance : photos de peau sans dermatoscope (12 novembre 2026)

Plan fixé avant le calcul (`scripts/modality_router_skin_photo_stress.py`, commit
`8db6e67`) : têtes des protocoles 1 et 2 inchangées, appliquées aux 6 294 recadrages
SLICE-3D de l'étude de faisabilité (photos du corps entier prises par une caméra 3D, pas
de dermoscopie ; quatre centres ; jamais utilisés pour ajuster le routeur). Embeddings
DINOv2-B repris du cache de cette étude (même transformation, empreinte du modèle
vérifiée). [Résultats](../results/modality_router_skin_photo_stress.json).

| SLICE-3D, 6 294 images | Fond d'œil | Dermoscopie | Inconnu | Probabilité de fond d'œil maximale |
|---|---|---|---|---|
| Protocole 1 | 0 | 5 790 (92 %) | 504 (8 %) | 0,08 |
| Protocole 2 | 0 | 5 745 (91 %) | 549 (9 %) | 0,09 |

**Critère de sécurité atteint** : aucune photo de peau envoyée en fond d'œil (borne haute
de l'IC exact 0,06 %, seuil 0,5 %), dans les quatre centres et pour les lésions malignes
comme bénignes.

**Lecture fixée à l'avance** : la grande majorité part en « dermoscopie ». Le routeur
reconnaît la peau, mais **ne distingue pas une photo ordinaire d'une dermoscopie**. Le
modèle de peau ayant appris sur des dermoscopies (ISIC 2019), il recevrait ces photos sans
avertissement : le type de photo doit rester déclaré par l'utilisateur, ou être appris
par le routeur avec une classe dédiée (non fait).

Réserves : photos recadrées d'une caméra de cabinet, pas de vraies photos de téléphone ;
une seule source.


**Vues externes de l'œil : non testées.** Recherche du 12 novembre 2026 : aucune base
ouverte et vérifiable trouvée. CP-AnemiC (conjonctives au téléphone, Mendeley) a été
retirée par ses auteurs ; Eyes-defy-anemia (IEEE DataPort) n'affiche aucune licence ;
les jeux Roboflow ou Kaggle « cataracte » ont une licence ou un contenu non vérifiés. Le
test ne vaut la peine que si le routeur doit décider seul ; avec un type de photo
déclaré et un routeur qui avertit, cette limite reste acceptable.
