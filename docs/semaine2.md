# Semaine 2 : analyse des erreurs

Le modèle fait 7 erreurs sur 58 avis de test (2 négatifs classés positifs, 5 positifs classés négatifs), et toutes portent sur des avis mitigés, du type « Commande jamais reçue ... sauf que ensuite je recommande ».
Ces erreurs viennent du sac de mots : le modèle additionne les mots positifs et négatifs sans tenir compte de leur ordre, donc il ne voit pas le retournement qui donne pourtant l'étiquette (la seconde moitié de l'avis).
Pour les corriger, il faudrait une représentation qui comprend le contexte et l'ordre des mots, comme des embeddings (semaine 9).