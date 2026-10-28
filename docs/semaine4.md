# Semaine 4 : tests approfondis et qualité, notes du binôme

> Une à deux pages. Uniquement des chiffres issus de VOS exécutions.

## 1. Couverture

- Avant (étape A) : 91 % (TOTAL 69 lignes, 6 non exécutées), lignes jamais exécutées : `api/main.py` 49, 77-79 et 85-87.
- Pour chacune, le comportement qu'aucun test ne vérifiait :
  - ligne 49 : la réponse 503 quand le modèle n'est pas chargé ;
  - lignes 77-79 : la réponse 500 quand le modèle plante pendant la prédiction ;
  - lignes 85-87 : le lancement direct du fichier (bloc `__main__`), jamais exécuté sous pytest.
- Après (étape D) : … % ; lignes exclues de la mesure et pourquoi : …

## 2. Défauts trouvés

| Symptôme | Test qui l'a révélé | Correction |
| --- | --- | --- |
| … | … | … |

## 3. Seuil d'abstention

- Deux ou trois points de la courbe (`python src/courbe_seuil.py`) : …
- Seuil retenu : … ; part des avis envoyés à un humain : … ; justification : …

## 4. Jeu d'or

- Bilan (`python src/bilan_jeu_or.py`) : justes …, abstentions …, erreurs franches déclarées …
- Ce que ces chiffres disent du modèle : …

## 5. Outils statiques

- Remarques au départ : ruff 15 (B904 ×2 dans `api/main.py` ; I001 et F401 dans `api/model.py` ; F401 et B905 ×2 dans `src/evaluate.py` ; I001 dans `tests/test_api.py` ; E741 ×3 dans `tests/test_cadrage.py` et `tests/test_environment.py` ; I001 dans `ui/app.py` et E741 ×3 dans `ui/helpers.py`, venant de l'ancienne interface), mypy : aucune remarque (Success, 4 fichiers sources).
- Parmi les défauts de la partie 2, combien ruff et mypy avaient-ils signalés ? Qu'en concluez-vous ? …