#!/usr/bin/env python3
"""
Simple spaCy test
"""

def run_spacy_diagnostics() -> bool:
    print("🔍 Testing spaCy Models")

    try:
        import spacy
        print(f"✅ spaCy installed, version: {spacy.__version__}")

        import spacy.util

        models = list(spacy.util.get_installed_models())
        print(f"📋 Installed models: {models}")

        test_models = ["en_core_web_sm", "zh_core_web_sm", "en_core_web_trf", "zh_core_web_trf"]

        for model in test_models:
            try:
                spacy.load(model)
                print(f"✅ {model} works")
            except Exception as exc:
                print(f"❌ {model} failed: {exc}")
        return True
    except Exception as exc:
        print(f"❌ spaCy error: {exc}")
        return False


if __name__ == "__main__":
    run_spacy_diagnostics()
