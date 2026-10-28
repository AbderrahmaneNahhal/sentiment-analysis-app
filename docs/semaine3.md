# \# Semaine 3 : API FastAPI

# 

# \## 1. Pourquoi l'API refuse-t-elle de démarrer sans modèle plutôt que de répondre 503 ?

# 

# Quand j'ai renommé models/sentiment.joblib puis lancé uvicorn, l'API s'est arrêtée

# avant d'écouter la moindre requête : FileNotFoundError "models/sentiment.joblib absent :

# lancez python src/train.py", suivi de "Application startup failed. Exiting."

# La trace montre que l'erreur vient de la fonction lifespan de api/main.py, qui charge

# le modèle au démarrage (app.state.classifier = load\_model()).

# 

# Refuser de démarrer est préférable à un 503 pour trois raisons :

# \- L'erreur est visible tout de suite, au déploiement, avec un message qui dit quoi faire,

# &#x20; au lieu d'apparaître plus tard chez un client.

# \- Une API qui démarre sans modèle semble "vivante" (le port répond, /docs s'affiche)

# &#x20; mais échoue à chaque appel à /predict : le problème est caché.

# \- Un orchestrateur ou une CI voit un processus qui s'arrête, donc un échec clair,

# &#x20; alors qu'un processus qui tourne en renvoyant 503 passe pour sain.

# 

# Le 503 reste utile pour un modèle qui disparaît pendant que l'API tourne,

# mais pas pour un modèle qui n'a jamais été chargé.

# 

# \## 2. Un texte vide est refusé deux fois : est-ce redondant ?

# 

# Au test T18, {"text": ""} reçoit un 422 de Pydantic (validation du schéma), avant

# même d'arriver au modèle. predict() du TP2 refuse aussi un texte vide.

# 

# Ce n'est pas une vraie redondance : ce sont deux protections à deux niveaux.

# \- Pydantic protège l'API : il rejette les requêtes mal formées à l'entrée, avec une

# &#x20; réponse 422 normalisée.

# \- predict() protège la fonction : elle est aussi appelée hors de l'API

# &#x20; (tests, scripts, notebook) et ne doit pas dépendre d'un schéma qu'elle ne voit pas.

# 

# Les deux protections ne couvrent pas les mêmes cas. À l'essai dans /docs, {"text": ""}

# reçoit un 422 dont detail est une liste d'objets (type string\_too\_short, loc body/text) :

# c'est Pydantic. Mais {"text": "   "} (trois espaces) passe le schéma, car la chaîne

# n'est pas vide, et c'est predict() qui le refuse : le 422 porte alors une simple chaîne,

# "texte vide". Un même refus a donc deux formes selon la couche qui le détecte. La

# double vérification est utile, car predict() sert aussi hors de l'API, mais cette

# incohérence de format est un défaut à corriger.

