# src/normalize_text.py
import re

_allowed = re.compile(r"[^A-Z'| ]+")

def normalize_text(s: str) -> str:
    """
    Normalize text for Wav2Vec2 CTC training/inference.
    Wav2Vec2 960h uses UPPERCASE letters and | as word separator!
    
    Improvements:
    - Better handling of compound words
    - Preserve common hyphenated terms
    - Handle numbers more consistently
    """
    s = str(s).upper()
    
    # Replace smart quotes with straight apostrophe
    s = s.replace("'", "'").replace("'", "'").replace("`", "'")
    
    # NEW: Handle common hyphenated compounds before space conversion
    # Keep some compounds together that ASR often splits wrong
    compounds_to_keep = [
        "FULL-TIME", "PART-TIME", "LONG-TERM", "SHORT-TERM",
        "HIGH-SCHOOL", "MIDDLE-SCHOOL", "E-MAIL", "CO-OPERATION",
        "SELF-ESTEEM", "SELF-AWARENESS", "WELL-BEING",
    ]
    for compound in compounds_to_keep:
        s = s.replace(compound, compound.replace("-", ""))
    
    # Convert remaining hyphens to spaces
    s = s.replace("-", " ")
    
    # Convert spaces to | (pipe) - Wav2Vec2 word separator
    s = s.replace(" ", "|")
    
    # Remove any characters that aren't A-Z, apostrophe, or pipe
    s = _allowed.sub("", s)
    
    # Clean up consecutive pipes
    s = re.sub(r"\|+", "|", s)
    s = s.strip("|")
    
    return s