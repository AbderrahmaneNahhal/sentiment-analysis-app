"""Tests aux limites du contrat d'entrée (TP4, étape B).

PredictRequest promet un champ « text », chaîne de 1 à 2000 caractères.
Ce fichier vérifie cette promesse à ses frontières, puis au hasard (hypothesis).
"""
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

MODELE = Path(__file__).resolve().parents[1] / "models" / "sentiment.joblib"
pytestmark = pytest.mark.skipif(not MODELE.exists(), reason="modèle non produit : python src/train.py")

ETIQUETTES = {"POSITIVE", "NEGATIVE", "INDETERMINE"}
# Blancs que str.strip() retire, y compris les inhabituels (\x85, espace insécable...)
BLANCS = " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0\u2003\u2028\u3000"


@pytest.fixture(scope="module")
def client():
    """Portée « module » : l'API démarre une fois pour tout le fichier (hypothesis l'exige)."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:  # le « with » exécute le lifespan : le modèle est chargé
        yield c


def en_json(valeur: object) -> bytes:
    """Corps de requête JSON, sous forme d'octets : permet aussi d'envoyer un corps invalide."""
    return json.dumps(valeur).encode("utf-8")


def verifier_forme_422(reponse) -> None:
    """detail doit être la liste d'erreurs standard de FastAPI : objets avec loc, msg, type."""
    detail = reponse.json()["detail"]
    assert isinstance(detail, list), f"detail devrait être une liste, reçu : {detail!r}"
    assert detail, "detail est une liste vide"
    for erreur in detail:
        assert {"loc", "msg", "type"} <= set(erreur), f"clés manquantes dans : {erreur!r}"


# Étape B.1 : huit cas, chacun avec son code attendu.
CAS = [
    pytest.param(en_json({"text": ""}), 422, id="texte-vide"),
    pytest.param(en_json({"text": "   "}), 422, id="espaces-seuls"),
    pytest.param(en_json({"text": "a"}), 200, id="un-caractere"),
    pytest.param(en_json({"text": "a" * 2000}), 200, id="2000-caracteres"),
    pytest.param(en_json({"text": "a" * 2001}), 422, id="2001-caracteres"),
    pytest.param(en_json({}), 422, id="champ-absent"),
    pytest.param(en_json({"text": 42}), 422, id="champ-nombre"),
    pytest.param(b"pas du json", 422, id="corps-non-json"),
]


@pytest.mark.parametrize("corps, code_attendu", CAS)
def test_frontieres_du_contrat(client, corps, code_attendu):
    reponse = client.post("/predict", content=corps, headers={"Content-Type": "application/json"})
    assert reponse.status_code == code_attendu
    # Étape B.2 : la forme de chaque 422
    if code_attendu == 422:
        verifier_forme_422(reponse)


# Étape B.5 : deux propriétés avec hypothesis.
@settings(deadline=None)
@given(texte=st.text(min_size=1, max_size=2000).filter(lambda t: t.strip()))
def test_propriete_texte_non_blanc(client, texte):
    """Tout texte contenant un caractère non blanc reçoit 200, une étiquette du contrat, un score dans [0, 1]."""
    reponse = client.post("/predict", json={"text": texte})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["label"] in ETIQUETTES
    assert 0.0 <= corps["score"] <= 1.0


@settings(deadline=None)
@given(texte=st.text(alphabet=BLANCS, min_size=1, max_size=200))
def test_propriete_texte_de_blancs(client, texte):
    """Tout texte fait uniquement de blancs reçoit 422, sous la forme standard."""
    reponse = client.post("/predict", json={"text": texte})
    assert reponse.status_code == 422
    verifier_forme_422(reponse)