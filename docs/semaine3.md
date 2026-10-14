# Semaine 3 : seuil d'incertitude

Nous retenons un seuil d'incertitude de 60 % : sous cette confiance, l'interface affiche « à vérifier » au lieu d'un verdict tranché.
Sur le jeu de démonstration (290 avis), cela envoie 117 avis (40 %) à un agent et rattrape 34 des 89 erreurs du modèle, mais des erreurs faites avec une confiance élevée, comme un avis positif classé négatif à 74 %, passent quand même.
Ce compromis nous paraît juste parce qu'un seuil de 75 % enverrait 81 % des avis à un agent pour rattraper 72 erreurs, ce qui ôte presque tout l'intérêt de l'automatisation, alors que 50 % ne filtre rien.