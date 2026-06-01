import os
import json
import threading

# Lazy loading of huggingface transformers
transformers = None
torch = None

# Paths
BASE_DIR = os.path.dirname(__file__)
CACHE_FILE = os.path.join(BASE_DIR, "cache.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
LANG_DIR = os.path.join(BASE_DIR, "lang")

# Globals for the pipeline
_model = None
_tokenizer = None
_model_lock = threading.Lock()
_cache_lock = threading.Lock()

# Persistent cache layer
_cache = {}

def get_available_languages():
    """Return the available languages for the user interface."""
    return ["eng_Latn", "hin_Deva", "pan_Guru", "tam_Taml"]

def load_config():
    """Loads the stored GUI configuration (like selected language)."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"language": "eng_Latn"}

def save_config(config):
    """Saves the GUI configuration."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"[WARN] Failed to save config: {e}")

def load_cache():
    """Loads the active translation cache."""
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    else:
        _cache = {}

def save_cache():
    """Saves the active translation cache to disk."""
    global _cache
    try:
        with _cache_lock:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_cache, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[WARN] Failed to save cache: {e}")

def load_language_json(lang_code):
    """Loads a specific language mapping for hardcoded UI elements."""
    json_path = os.path.join(LANG_DIR, f"{lang_code}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback to English
    fallback_path = os.path.join(LANG_DIR, "eng_Latn.json")
    if os.path.exists(fallback_path):
        with open(fallback_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_model():
    """Lazy initialization of the IndicTrans2 model into RAM unconditionally wrapped by threading lock."""
    global _model, _tokenizer, transformers, torch
    with _model_lock:
        if _model is not None:
            return # Already loaded
            
        print("[INFO] Initializing offline IndicTrans2 Translation Engine...")
        try:
            import torch as _torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            torch = _torch
            transformers = True
            
            model_name = "ai4bharat/indictrans2-en-indic-1B"
            
            # Ensure initialization runs strictly on CPU mapping
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
            _model.to("cpu")
            _model.eval()
            
            print("[INFO] Translation Engine successfully loaded into Memory.")
        except Exception as e:
            print(f"[ERROR] Failed to load offline translation model: {e}")
            _model = None
            _tokenizer = None

def translate_text(text, target_lang):
    """Deep translate execution through IndicTrans2 interface."""
    global _model, _tokenizer
    if not text or not text.strip():
        return text
    
    if target_lang == "eng_Latn" or "eng" in target_lang.lower():
        return text
        
    if _model is None or _tokenizer is None:
        load_model()
        
    if _model is None:
        return text # Ultimate fallback if model loading crashed permanently
        
    try:
        with _model_lock:
            # Note: We must specify the token formats appropriately for IndicTrans2.
            # Usually requires direction wrapper depending on library version. 
            # In purely vanilla usages, encode standard inputs:
            inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cpu")
            
            with torch.no_grad():
                generated_tokens = _model.generate(
                    **inputs,
                    use_cache=True, 
                    min_length=0, 
                    max_length=256,
                    num_beams=1, 
                )
            
            # Decode payload
            translated = _tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated
    except Exception as e:
        print(f"[WARN] Model translation failed: {e}")
        return text

def smart_translate(text, target_lang):
    """
    Cached facade wrapping translation functionality to execute fast lookup
    before deferring to transformer evaluation logic.
    """
    if not text or not text.strip() or target_lang == "eng_Latn":
        return text
        
    cache_key = f"{text}_{target_lang}"
    
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]
            
    # Buffer cache-miss to backend processing
    translated = translate_text(text, target_lang)
    
    # Store evaluated target into global table
    with _cache_lock:
        _cache[cache_key] = translated
    
    # Fire and forget async save to prevent disk-IO freezing
    threading.Thread(target=save_cache, daemon=True).start()
    
    return translated

# Execute fundamental operations unconditionally at file load
load_cache()
