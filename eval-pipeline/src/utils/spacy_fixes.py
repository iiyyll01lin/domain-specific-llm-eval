#!/usr/bin/env python3
"""
spaCy Fixes for Chinese Language Support
"""


def get_contextual_segments_fixed(text, language="english"):
    """
    Fixed version that handles Chinese language properly
    """
    import spacy

    # Load appropriate model
    if language == "chinese":
        try:
            nlp = spacy.load("zh_core_web_sm")
        except:
            # Fallback to simple segmentation for Chinese
            return _chinese_text_segmentation(text)
    else:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback to simple segmentation
            return _simple_text_segmentation(text)
    
    doc = nlp(text.lower())
    segments = []
    
    # Get named entities (works for both languages)
    segments.extend([ent.text for ent in doc.ents])
    
    # Language-specific processing
    if language == "chinese":
        # For Chinese, use token-based segmentation instead of noun_chunks
        segments.extend([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 1])
        
        # Add sentence-level segments
        segments.extend([sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5])
    else:
        # For English, use noun_chunks safely
        try:
            segments.extend([chunk.text for chunk in doc.noun_chunks])
        except:
            # Fallback if noun_chunks fails
            segments.extend([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]])
    
    # Get dependency-based phrases (works for both languages)
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ["ROOT", "xcomp", "ccomp", "advcl"]:
                phrase = " ".join([t.text for t in token.subtree])
                segments.append(phrase)
    
    # Clean and deduplicate
    segments = list(set([seg.strip() for seg in segments if len(seg.split()) > 1]))
    return segments

def _chinese_text_segmentation(text):
    """Simple Chinese text segmentation fallback"""
    import re

    # Split by Chinese punctuation and spaces
    segments = re.split(r'[，。！？；：、\s]+', text)
    segments = [seg.strip() for seg in segments if len(seg.strip()) > 2]
    return segments

def _simple_text_segmentation(text):
    """Simple English text segmentation fallback"""
    import re

    # Split by punctuation and extract meaningful phrases
    sentences = re.split(r'[.!?;]+', text)
    segments = []
    
    for sentence in sentences:
        # Extract noun-like phrases (simple heuristic)
        words = sentence.split()
        for i, word in enumerate(words):
            if word.lower() in ['the', 'a', 'an'] and i + 1 < len(words):
                # Extract noun phrase starting with article
                phrase_words = [words[i]]
                for j in range(i + 1, min(i + 4, len(words))):
                    if words[j].isalpha():
                        phrase_words.append(words[j])
                    else:
                        break
                if len(phrase_words) > 1:
                    segments.append(' '.join(phrase_words))
    
    return segments
