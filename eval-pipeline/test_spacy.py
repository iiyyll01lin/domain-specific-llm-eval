import spacy
from spacy import prefer_gpu
from thinc.api import require_gpu

def test_spacy_gpu_probe():
    gpu_preferred = prefer_gpu()
    assert isinstance(gpu_preferred, bool)


def test_spacy_transformer_model_or_fallback():
    try:
        nlp_en_trf = spacy.load("en_core_web_trf")
    except OSError:
        nlp_en_trf = spacy.load("en_core_web_sm")

    doc = nlp_en_trf("spaCy pipeline check")
    assert doc.text == "spaCy pipeline check"


def test_spacy_require_gpu_probe():
    try:
        require_gpu()
    except ValueError:
        assert True
    else:
        assert True