"""Cleaning helpers for the trial data transfer."""
import difflib
import re

import numpy as np
import pandas as pd

# alternative unit -> (standard unit, factor to multiply by)
UNIT_FACTORS = {
    ("GLUC", "mg/dL"): ("mmol/L", 18.016),
    ("CREAT", "mg/dL"): ("umol/L", 88.42),
    ("HGB", "g/dL"): ("g/L", 10.0),
    ("ALB", "g/dL"): ("g/L", 10.0),
    ("CHOL", "mg/dL"): ("mmol/L", 1 / 38.67),
    ("TRIG", "mg/dL"): ("mmol/L", 1 / 88.57),
    ("HDL", "mg/dL"): ("mmol/L", 1 / 38.67),
    ("LDL", "mg/dL"): ("mmol/L", 1 / 38.67),
}

ARMS = {
    "placebo": "Placebo", "pbo": "Placebo",
    "low dose": "Low dose", "drug 10mg": "Low dose", "low": "Low dose",
    "high dose": "High dose", "drug 40mg": "High dose", "high": "High dose",
}

PREFERRED_TERMS = {
    "headache": "Headache", "head ache": "Headache",
    "nausea": "Nausea", "feeling sick": "Nausea",
    "fatigue": "Fatigue", "tiredness": "Fatigue", "tired": "Fatigue",
    "dizziness": "Dizziness", "dizzy": "Dizziness",
    "diarrhoea": "Diarrhoea", "diarrhea": "Diarrhoea", "loose stools": "Diarrhoea",
    "rash": "Rash", "skin rash": "Rash",
    "insomnia": "Insomnia", "sleeplessness": "Insomnia", "trouble sleeping": "Insomnia",
    "alt increased": "ALT increased", "elevated alt": "ALT increased", "alt elevation": "ALT increased",
    "hypoglycaemia": "Hypoglycaemia", "hypoglycemia": "Hypoglycaemia", "low blood sugar": "Hypoglycaemia",
    "back pain": "Back pain", "backache": "Back pain",
    "nasopharyngitis": "Nasopharyngitis", "common cold": "Nasopharyngitis", "cold": "Nasopharyngitis", "urti": "Nasopharyngitis",
    "arthralgia": "Arthralgia", "joint pain": "Arthralgia",
}

SOC = {
    "Headache": "Nervous system", "Dizziness": "Nervous system", "Nausea": "Gastrointestinal",
    "Diarrhoea": "Gastrointestinal", "Fatigue": "General", "Rash": "Skin", "Insomnia": "Psychiatric",
    "ALT increased": "Investigations", "Hypoglycaemia": "Metabolism", "Back pain": "Musculoskeletal",
    "Arthralgia": "Musculoskeletal", "Nasopharyngitis": "Infections",
}


def normalize_arm(s):
    return s.str.strip().str.lower().map(ARMS)


def normalize_sex(s):
    return s.str.strip().str[0].str.upper()


def standardize_units(labs):
    """Convert every result to the standard unit of its test."""
    out = labs.copy()
    out["value_std"] = out["value"]
    out["unit_std"] = out["unit"]
    for (test, unit), (std_unit, factor) in UNIT_FACTORS.items():
        m = (out["test_code"] == test) & (out["unit"] == unit)
        out.loc[m, "value_std"] = out.loc[m, "value"] * factor
        out.loc[m, "unit_std"] = std_unit
    return out


def drop_retests(labs):
    """Keep the last result per subject, visit and test (re-tests replace the first draw)."""
    return labs.drop_duplicates(["subject_id", "visit", "test_code"], keep="last").reset_index(drop=True)


def flag_range(labs, ref):
    out = labs.merge(ref.rename(columns={"unit": "unit_ref"}), on="test_code", how="left")
    out["flag"] = np.select([out["value_std"] < out["low"], out["value_std"] > out["high"]], ["L", "H"], "N")
    return out.drop(columns=["unit_ref"])


def normalize_ae_term(verbatim):
    """Map a verbatim AE term to a preferred term (dictionary first, fuzzy match second)."""
    if not isinstance(verbatim, str):
        return "Uncoded"
    t = re.sub(r"\(.*?\)", "", verbatim).strip().lower()
    t = re.sub(r"\s+", " ", t)
    if t in PREFERRED_TERMS:
        return PREFERRED_TERMS[t]
    for key in PREFERRED_TERMS:
        if key in t:
            return PREFERRED_TERMS[key]
    close = difflib.get_close_matches(t, list(PREFERRED_TERMS), n=1, cutoff=0.75)
    return PREFERRED_TERMS[close[0]] if close else "Uncoded"


def severity_grade(s):
    s = s.astype(str).str.strip().str.lower()
    return s.map({"mild": 1, "1": 1, "moderate": 2, "2": 2, "severe": 3, "3": 3})
