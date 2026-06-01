import sys
import os

def get_multilingual_output(text, lang_code="eng_Latn"):
    """
    Translates robust diagnostic texts.
    Submits FULL strings rather than chunking mapped IDs ensuring context holds.
    """
    if lang_code == "eng_Latn":
        return text
        
    try:
        from gui.language_manager import smart_translate
        # We pass the ENTIRE block as explicitly bounded so that the neural model 
        # maintains contextual mappings for the ID logic.
        translated_block = smart_translate(text, lang_code)
        return translated_block
    except Exception as e:
        print(f"[WARN] Aadhaar multilingual translation failed: {e}")
        return text
