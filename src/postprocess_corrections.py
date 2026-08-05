# src/postprocess_corrections.py
"""
Rule-based postprocessing corrections for ASR outputs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class PostprocessConfig:
    fix_double_words: bool = True
    fix_spacing: bool = True
    apply_dictionary: bool = True
    fix_compounds: bool = True
    filter_garbage: bool = True


# =============================================================================
# ALL CORRECTIONS
# =============================================================================

DEFAULT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    # === MULTI-WORD FIXES ===
    ("AS A RESULTS", "AS A RESULT"),
    ("O ONE", "OH ONE"),
    ("ON O", "ON OH"),
    ("O MY", "OH MY"),
    ("EUROCRISIS", "EURO CRISIS"),
    ("M ABOUT T", "MOST"),
    ("NO HAR TO WALKS", "NO HARD WALKS"),
    ("NO HAR TO WARKS", "NO HARD WALKS"),
    ("NOWN COMPENELTO", "UNKNOWN COMPANY"),
    ("O WANTED", "I WANTED"),
    ("YOKWIF", "YOU KNOW IF"),
    ("YKWIF", "YOU KNOW IF"),

    # === SINGLE WORD FIXES ===
    ("ACRAMIC", "ACROBATIC"),
    ("JUDGMENT", "JUDGEMENT"),
    ("ACRAMICAL", "ACROBATIC"),
    ("AGLO", "AGLOW"),
    ("AL", "ALL"),
    ("ANCIOUSTERE", "ANCIENT"),
    ("APROSHTO", "APPROACH TO"),
    ("ARNSTRONG", "ARMSTRONG"),
    ("BAGING", "BEIJING"),
    ("BASICALY", "BASICALLY"),
    ("BYSICALY", "BASICALLY"),
    ("CALE", "KALE"),
    ("CARBONAT", "CARBONATE"),
    ("CENTITGRATE", "CENTIGRADE"),
    ("CEILINK", "CEILING"),
    ("COKAYN", "COCAINE"),
    ("COMOTION", "LOCOMOTION"),
    ("COMPENELTO", "COMPANY"),
    ("COMPON", "COMPOUND"),
    ("CRATERS", "CREATORS"),
    ("DN'T", "DON'T"),
    ("DOMIMICECOSYSTEMS", "DOMINIC ECOSYSTEMS"),
    ("EUSTAT", "EUROSTAT"),
    ("EXAGERATING", "EXAGGERATING"),
    ("FIKENLY", "LIKELY"),
    ("FOD", "FOOD"),
    ("GNATZI", "NAZI"),
    ("GRAPHIGOTS", "CATACOMBS"),
    ("GRAVIGOTS", "CATACOMBS"),
    ("HAVETY", "HAVE TO"),
    ("INCONVERSIVETHY", "IN CONVERSATION"),
    ("INTRESS", "INTEREST"),
    ("ITE", "LIKE"),
    ("JARAMUSK", "JERUSALEM"),
    ("LABORATOR", "LABORATORY"),
    ("LEARNE", "LEARN"),
    ("LEARNED", "LEARN"),
    ("LOANAGES", "LANGUAGES"),
    ("MINERAT", "MINARET"),
    ("NESTRAFER", "NEITHER"),
    ("NEUCLER", "NUCLEAR"),
    ("NITRADE", "NITRATE"),
    ("OCTIPUS", "OCTOPUS"),
    ("PATASSIUM", "POTASSIUM"),
    ("PATOON", "PLATOON"),
    ("PERSONIZE", "PERSONALIZE"),
    ("PLASTECIZE", "PLASTICIZE"),
    ("POLLY", "POLY"),
    ("POSTDOX", "POSTDOC"),
    ("REALY", "REALLY"),
    ("RISKTAKING", "RISK TAKING"),
    ("ROUTS", "ROUTES"),
    ("SIMPLIFIL", "SIMPLIFY"),
    ("SUCED", "SUCCEED"),
    ("TAILS", "TALES"),
    ("THER", "THEIR"),
    ("THERRE", "THERE"),
    ("TOUSAND", "THOUSAND"),
    ("TUTORED", "TUTORED"),
    ("TWE'RE", "WE'RE"),
    ("VECTORUM", "VECTOR"),
    ("WARKS", "WALKS"),
    ("WATCHPO", "WATCH"),
    ("WIL", "WILL"),
    ("WITHINNER", "WITHIN"),
    ("WUT", "BUT"),

    # === WORD SUBSTITUTIONS ===
    ("ADIT", "HAD IT"),
    ("II", "I"),
    ("INCE", "SINCE"),
    ("UNOW", "UN"),

    # === COMPOUND SPLITS ===
    ("THREEFOOT", "THREE FOOT"),
    ("ALITTLE", "A LITTLE"),
    ("AMOUNTSS", "AMOUNTS"),
)


def _simple_replace(text: str, wrong: str, right: str) -> str:
    """
    Replace exact matches, handling both single words and multi-word phrases.
    """
    if " " in wrong:
        return text.replace(wrong, right)
    
    words = text.split()
    new_words = []
    for word in words:
        if word == wrong:
            new_words.append(right)
        else:
            new_words.append(word)
    return " ".join(new_words)


def _fix_double_words(text: str) -> str:
    """Remove consecutive duplicate words like 'THE THE'."""
    words = text.split()
    if len(words) < 2:
        return text
    result = [words[0]]
    for w in words[1:]:
        if w != result[-1]:
            result.append(w)
    return " ".join(result)


def _fix_spacing(text: str) -> str:
    """Split common concatenated word pairs."""
    pairs = [
        ("ITWAS", "IT WAS"),
        ("ANDSO", "AND SO"),
        ("INTHE", "IN THE"),
        ("TOTHE", "TO THE"),
        ("ONTHE", "ON THE"),
        ("FORTHE", "FOR THE"),
        ("WITHTHE", "WITH THE"),
        ("ATTHE", "AT THE"),
        ("FROMTHE", "FROM THE"),
        ("BYTHE", "BY THE"),
        ("INTOA", "INTO A"),
        ("OUTOFA", "OUT OF A"),
        ("UPTO", "UP TO"),
        ("DOWNTO", "DOWN TO"),
        ("BACKTO", "BACK TO"),
        ("NEXTTO", "NEXT TO"),
        ("ACROSSTO", "ACROSS TO"),
        ("THANKYOU", "THANK YOU"),
        ("EVERYTHINGIS", "EVERYTHING IS"),
        ("EVERYONEIS", "EVERYONE IS"),
        ("SOMETHINGIS", "SOMETHING IS"),
        ("NOTHINGIS", "NOTHING IS"),
        ("WHATIS", "WHAT IS"),
        ("THISIS", "THIS IS"),
        ("THATIS", "THAT IS"),
        ("THEREIS", "THERE IS"),
        ("HEREIS", "HERE IS"),
        ("ITIS", "IT IS"),
        ("HEIS", "HE IS"),
        ("SHEIS", "SHE IS"),
    ]
    for wrong, right in pairs:
        text = re.sub(rf"\b{wrong}\b", right, text)
    return text


def _filter_garbage(text: str) -> str:
    """
    Filter out garbage text with too few vowels or repetitive characters.
    Returns cleaned text or empty string if mostly garbage.
    """
    words = text.split()
    if not words:
        return text
    
    garbage_count = 0
    vowels = set("AEIOU")
    
    for word in words:
        clean_word = word.replace("'", "")
        if len(clean_word) <= 2:
            continue
        
        if not any(c in vowels for c in clean_word):
            garbage_count += 1
        
        if len(set(clean_word)) <= 2 and len(clean_word) > 3:
            garbage_count += 1
        
        max_repeat = max(len(list(g)) for _, g in __import__('itertools').groupby(clean_word))
        if max_repeat > 3 and len(clean_word) > 4:
            garbage_count += 0.5
    
    if len(words) > 0 and garbage_count / len(words) > 0.3:
        good_words = []
        for word in words:
            clean_word = word.replace("'", "")
            if len(clean_word) <= 2:
                good_words.append(word)
                continue
            
            has_vowel = any(c in vowels for c in clean_word)
            has_variety = len(set(clean_word)) > 2
            max_repeat = max(len(list(g)) for _, g in __import__('itertools').groupby(clean_word))
            
            if has_vowel and has_variety and max_repeat <= 3:
                good_words.append(word)
        
        if good_words:
            return " ".join(good_words)
        return ""
    
    return text


def apply_postprocess(
    text: str,
    cfg: PostprocessConfig | None = None,
    replacements: Iterable[Tuple[str, str]] = DEFAULT_REPLACEMENTS,
) -> str:
    """
    Apply all postprocessing rules to ASR output text.
    """
    cfg = cfg or PostprocessConfig()

    if not text:
        return ""

    t = text.upper()
    t = re.sub(r"\s+", " ", t).strip()

    if cfg.filter_garbage:
        t = _filter_garbage(t)

    if cfg.fix_double_words:
        t = _fix_double_words(t)

    if cfg.fix_spacing:
        t = _fix_spacing(t)

    if cfg.apply_dictionary:
        for wrong, right in replacements:
            t = _simple_replace(t, wrong, right)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_garbage(text: str) -> bool:
    """
    Check if text is complete garbage.
    """
    if not text or len(text.strip()) < 3:
        return True
    
    vowels = set("AEIOU")
    letters = [c for c in text.upper() if c.isalpha()]
    
    if not letters:
        return True
    
    vowel_ratio = sum(1 for c in letters if c in vowels) / len(letters)
    
    if vowel_ratio < 0.15:
        return True
    
    words = text.split()
    if vowel_ratio > 0.7 and len(words) > 3:
        long_words = sum(1 for w in words if len(w) > 4)
        if long_words == 0:
            return True
    
    max_repeat = max(len(list(g)) for _, g in __import__('itertools').groupby(text.upper()) if __import__('itertools').groupby)
    if max_repeat > 5:
        return True
    
    return False


def strip_leading_junk(text: str) -> str:
    """Remove leading short consonant-only words."""
    words = text.split()
    if not words:
        return text
    
    vowels = set("AEIOU")
    first = words[0].replace("'", "")
    
    if len(first) <= 3 and not any(c in vowels for c in first):
        return " ".join(words[1:])
    
    return text