# Compte rendu global des TP1 à TP3

**Équipe (trio) :** Othmane TOUITAT KABBAJ ([@othmanekt0]), Mohammed BARRA ([@mohammedbarra2005-arch]) et Abderrahmane NAHHAL ([@AbderrahmaneNahhal])
**Dépôt :** https://github.com/othmanekt0/TOUITAT_KABBAJ_OTHMANE-A--BARRA_Mohammed-B--NAHHAL_Abderrahmane-C
**Date :** 14/10/2026

## 1. Le besoin et son cadrage

Une équipe relation client reçoit chaque jour plusieurs milliers d'avis. Elle répond en priorité aux clients mécontents, mais lire les avis dans leur ordre d'arrivée fait attendre les négatifs derrière les positifs. Le besoin est donc de classer automatiquement les avis en positifs et négatifs, pour faire remonter les négatifs en tête de la file.

Pour ce cas 3, nous avons retenu le **machine learning** (un classifieur entraîné sur des avis étiquetés), et non des règles métier ni un modèle génératif. Le raisonnement suit la grille de décision :

- **(a) Données étiquetées.** Les avis s'étiquettent presque gratuitement à partir de la note en étoiles que le client a déjà saisie. C'est la condition qui rend l'apprentissage supervisé possible.
- **(c) Coût par requête.** Plusieurs milliers d'avis par jour : un classifieur coûte une fraction de milliseconde par avis, alors qu'un modèle de langage se paierait à chaque appel.
- **(d) Conséquence d'une erreur.** Un avis négatif classé positif est traité quelques heures plus tard et l'agent le requalifie d'un clic. L'erreur n'est pas irréversible, ce qui autorise une automatisation partielle.

Nous avons écarté les règles par mots-clés : la négation et l'ironie leur échappent (« rien à redire » serait compté comme négatif). La métrique de succès fixée dans la fiche est le **rappel sur la classe négative, avec un seuil d'acceptation de 0,90**, parce que manquer un client mécontent coûte plus cher que relire un avis positif.

## 2. L'architecture du dépôt

- `data/` : les jeux d'avis. Seul `sample_reviews.csv` (10 avis d'exemple) est versionné. `demo_reviews.csv` (290 avis) est régénéré par `src/make_demo.py` avec une graine fixe, donc il n'a pas besoin d'être dans Git.
- `src/` : les scripts du TP2 (`make_demo.py`, `train.py`, `evaluate.py`, `predict.py`).
- `models/` : les sorties de l'entraînement.
- `ui/` : l'interface Streamlit du TP3 (`app.py` pour l'affichage, `helpers.py` pour les fonctions pures et le seuil).
- `app/` : réservé pour la suite du cours, vide pour l'instant.
- `tests/` : 45 tests `pytest` (38 du TP1, 4 du TP2, 3 du TP3).
- `scripts/` et `.github/workflows/ci.yml` : le diagnostic d'installation et l'intégration continue.
- `docs/` : les fiches de cadrage (`docs/cadrage/`), `semaine2.md`, `semaine3.md` et ce compte rendu.
- `notebooks/` : les notebooks Colab des trois séances.

**Ce qui est versionné et ce qui ne l'est pas.** Le `.gitignore` exclut `models/*`, avec une exception explicite `!models/metrics.json`. Le fichier `models/sentiment.joblib` n'est pas dans Git pour trois raisons : c'est un binaire lourd, il ne se relit pas dans un diff, et il se reconstruit à l'identique avec `python src/train.py`. À l'inverse, `models/metrics.json` est un petit fichier texte qui garde une trace de chaque entraînement (source, date, tailles, F1, rappel). Il sert aussi à la CI : les tests du TP2 lisent ce fichier pour vérifier que le F1 dépasse le seuil de 0,85, donc la CI peut valider les 45 tests sans disposer du modèle. Le `.venv/`, les secrets (`.env`) et `mlruns/` sont exclus pour des raisons voisines : ce qui se reconstruit ou ce qui est confidentiel ne va pas dans l'historique.

## 3. Le modèle

**Données.** Deux jeux. Un jeu de démonstration de 290 avis (145 positifs, 145 négatifs) généré par `make_demo.py`, dont 100 avis mitigés (de la forme « … sauf que ensuite … ») et 30 neutres, volontairement difficiles. Puis un vrai jeu public d'avis en français, Allociné : 20 000 avis, avec leur étiquette. Ce sont des critiques de films, pas des avis de commandes, nous y revenons en section 4.

**Découpage.** 80 % pour l'entraînement, 20 % pour le test, avec `train_test_split` stratifié (même proportion de positifs et de négatifs dans les deux parties) et une graine fixe. Le découpage est fait **avant** la vectorisation : le vocabulaire est appris uniquement sur l'entraînement. Si on vectorisait d'abord, le modèle connaîtrait déjà des mots du jeu de test, et le score mesuré serait trop optimiste.

**Pipeline.** `TfidfVectorizer` (mots et paires de mots, `min_df=2`, `sublinear_tf`) suivi d'une `LogisticRegression` avec `class_weight="balanced"`. Le texte devient un vecteur de pondérations de mots, et la régression apprend quels mots penchent vers positif ou négatif.

**Résultats** (lus dans nos sorties de terminal et dans `models/metrics.json`) :

| Jeu | Avis | Vocabulaire | F1 macro | Exactitude | Rappel négatif |
| --- | --- | --- | --- | --- | --- |
| Démonstration | 290 (test : 58) | 316 | 0,879 | 0,879 | 0,931 |
| Allociné | 20 000 (test : 4 000) | 149 529 | 0,9205 | 0,9205 | 0,9113 |

Les deux F1 dépassent le seuil de 0,85. Sur Allociné, le rappel négatif de 0,9113 dépasse aussi le seuil de 0,90 de notre fiche de cadrage. Le `metrics.json` versionné est celui d'Allociné, car c'est le dernier entraînement.

**Matrice de confusion du jeu de démonstration** (lignes = vérité, colonnes = prédiction) :

```text
            negative  positive
  negative        27         2
  positive         5        24
```

- Ligne « negative » : sur 29 vrais négatifs, 27 sont retrouvés et 2 sont classés positifs. D'où le rappel négatif de 27/29 = 0,931.
- Ligne « positive » : sur 29 vrais positifs, 24 sont retrouvés et 5 sont classés négatifs. Le rappel positif est de 24/29 = 0,828.
- Le modèle se trompe donc plus dans le sens « positif classé négatif ». Pour notre besoin, c'est le moins grave des deux : un agent relit un avis positif en trop, mais aucun client mécontent n'est oublié.

## 4. L'analyse des erreurs

Sur le jeu de démonstration, le modèle fait 7 erreurs sur 58 avis (2 négatifs classés positifs, 5 positifs classés négatifs). Elles portent **toutes** sur des avis mitigés. Sur Allociné, l'exactitude de 0,9205 sur 4 000 avis correspond à environ 318 erreurs (valeur déduite de l'exactitude, nous n'avons pas affiché la matrice complète).

**Le mécanisme.** Le TF-IDF compte les mots et les paires de mots, mais ne représente pas leur position dans la phrase. Dans un avis du type « problème au début, sauf que ensuite tout va bien », les mots négatifs sont aussi nombreux que les mots positifs, et le modèle ne sait pas que l'étiquette suit la seconde moitié. Ce qu'il lui faudrait, c'est une représentation qui tienne compte de l'ordre et du contexte.

**Exemples tirés de nos données.** Deux avis du jeu de démonstration, issus de `evaluate.py` :

- « Commande jamais reçue et remboursement en attente au début, sauf que ensuite je recommande » : vérité positive, prédit négatif.
- « Service impeccable et livraison en avance sur le papier, sauf que ensuite produit cassé… » : vérité négative, prédit positif.

Dans les deux cas, le début de l'avis porte des mots très marqués qui l'emportent sur la fin, alors que la fin décide de l'étiquette.

**Un second effet, découvert au TP3 : le décalage de domaine.** En faisant analyser `demo_reviews.csv` par l'interface, nous avons obtenu 89 erreurs sur 290 (31 %), contre 7 sur 58 au TP2. Ce n'est pas une contradiction : au TP2, le modèle de démonstration avait été entraîné sur ce même type d'avis, alors que l'interface utilisait le modèle Allociné, appris sur des critiques de films. Il surestime les négatifs (210 détectés sur 290, soit 72,4 %, pour 145 négatifs réels). Sur `sample_reviews.csv`, nous avons aussi observé 4 erreurs sur 10.

**Ce qu'il faudrait pour corriger.** Pour le mécanisme : des représentations qui tiennent compte de l'ordre des mots (embeddings de phrases ou modèle de type transformeur). Pour le décalage de domaine : entraîner ou ajuster sur de vrais avis clients étiquetés par leur note. Ce n'était pas l'objet de ces séances, dont le but était de mesurer un premier modèle simple et de comprendre ses limites avant de passer à des modèles plus riches (embeddings, vus plus tard dans le cours).

## 5. L'interface et le seuil d'incertitude

L'interface Streamlit a trois onglets, dont l'analyse d'un avis isolé et l'analyse d'un fichier CSV avec trois indicateurs (avis analysés, négatifs détectés, cas à vérifier), un graphique et un tableau avec les négatifs en premier. Quand la confiance du modèle est inférieure au seuil, elle affiche « à vérifier par un agent » au lieu d'un verdict tranché.

**Le seuil retenu : 0,60.** Chiffres mesurés sur `demo_reviews.csv` (290 avis, 89 erreurs du modèle actif) :

| Seuil | Avis envoyés à un agent | Erreurs rattrapées (sur 89) |
| --- | --- | --- |
| 0,50 | 0 (0 %) | 0 |
| 0,60 | 117 (40 %) | 34 |
| 0,75 | 234 (81 %) | 72 |

**L'arbitrage.** À 0,50, rien ne passe jamais sous le seuil (la confiance minimale d'un classifieur à deux classes est de 0,5) : l'interface tranche tout. À 0,75, l'agent rattrape 72 erreurs, mais il relit plus de 8 avis sur 10, donc l'automatisation ne sert presque plus. À 0,60, l'agent traite 40 % des avis et rattrape environ un tiers des erreurs. Pour l'équipe relation client, c'est un compromis acceptable : la charge de relecture reste raisonnable et les avis où le modèle hésite sont vus par un humain.

**La limite du seuil.** Il ne rattrape que les erreurs où le modèle hésite, pas celles où il est faussement sûr de lui. Nous l'avons constaté : l'avis « Taille trop petite et retour payant au déballage, sauf que ensuite produit conforme » est classé négatif avec 74 % de confiance alors qu'il est positif. De même, « Service impeccable au début, mais finalement produit cassé » est bien classé négatif, mais à 63 %, donc sans alerte. Un seuil ne remplace pas un meilleur modèle. Ces chiffres sont mesurés avec le modèle Allociné sur des avis de commandes, donc ils sont pessimistes : avec un modèle adapté au domaine, le nombre d'erreurs et le choix du seuil seraient à refaire.

## 6. Ce qui a échoué

- **Dépôt perdu.** Au début de la session, Othmane ne savait plus où se trouvait son dossier local. Nous avons recloné le dépôt dans `Documents\EMSI`, recréé le `.venv` et réinstallé les dépendances. Cela a coûté une dizaine de minutes (l'installation de `requirements.txt` est longue).
- **Mauvais dossier courant.** Lors de l'installation du kit TP2, la commande `cd` n'avait pas été exécutée : le terminal était encore dans le dépôt, et le `dir` ne montrait pas le contenu attendu. Nous avons remonté d'un niveau pour vérifier que `tp2-semaine2` était bien à côté du dépôt.
- **Tests rouges au contrôle C1.** Juste après l'installation du kit TP2, deux tests échouaient (`models/metrics.json absent`). C'était attendu : le fichier n'existe qu'après l'entraînement. Ces deux tests sont passés au vert dès l'étape `train.py`.
- **Identité Git incohérente.** Le diagnostic a listé 4 auteurs pour 3 personnes : le premier commit d'Othmane avait été fait avec un autre nom et un autre e-mail (`othmanekt0`). Git compte le couple nom + e-mail, pas la personne. Nous avons réglé `git config user.name` et `user.email` pour les commits suivants ; l'historique existant reste inchangé.
- **Message de commit mal rempli.** Un message de commit contient des chevrons `<Nahhal Abderrahmane>` laissés par un copier-coller. Sans effet sur les tests, mais pas propre.
- **Avertissement de version Python.** Le diagnostic signale Python 3.10 au lieu de 3.11. Sans conséquence sur les 45 tests.
- **Fausse piste sur le seuil.** Nous nous attendions à ce qu'un avis mitigé passe toujours en « à vérifier ». Le premier essai a donné un verdict net à 63 %, et un autre une erreur à 74 % : c'est ce qui nous a fait comprendre la limite du seuil (section 5).
- **Résultats trop différents entre TP2 et TP3.** Nous avons d'abord été surpris par les 89 erreurs de l'interface face aux 7 du TP2, avant de comprendre qu'il s'agissait de deux modèles différents (section 4).

## 7. Ce que nous ferions autrement

Avec les moyens d'une vraie équipe : (1) entraîner sur de vrais avis clients de l'entreprise, étiquetés par leur note, au lieu de critiques de films, et mesurer le rappel négatif sur un jeu de test de ce même domaine ; (2) essayer des embeddings de phrases pour traiter les avis mitigés ; (3) mesurer le seuil sur un jeu de validation, puis le tester en conditions réelles, en comptant sur la charge réelle des agents ; (4) harmoniser dès le premier jour la configuration Git des trois membres et le nom du dépôt ; (5) enregistrer les métriques de chaque entraînement séparément, pour ne pas écraser celles d'un jeu par celles d'un autre.

## 8. Répartition du travail

| Membre | Ce qu'il a pris en charge |
| --- | --- |
| Othmane TOUITAT KABBAJ | cadrage, mise en place du dépôt, TP2 (modèle, évaluation, analyse des erreurs), tests, CI, commits et push. |
| Mohammed BARRA | L'interface Streamlit (TP3) : onglets, affichage des verdicts et du seuil « à vérifier », import et export de fichiers CSV. |
| Abderrahmane NAHHAL | La rédaction du compte rendu. |

## Outils d'aide à la rédaction

Nous avons utilisé un assistant d'IA (Claude) pour nous guider pas à pas dans les commandes des TP2 et TP3 et pour préparer le brouillon de ce compte rendu à partir de nos sorties de terminal. Nous avons vérifié nous-mêmes que les chiffres correspondent à `models/metrics.json`, à nos sorties de `evaluate.py` et à nos exports de l'interface, que les 45 tests passent et que la CI est verte.
