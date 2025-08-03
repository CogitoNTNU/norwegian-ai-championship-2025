"""Medical text preprocessing utilities."""

import re
from typing import List


# Common medical abbreviations
MEDICAL_ABBREVIATIONS = {
    "pt": "patient",
    "pts": "patients",
    "w/": "with",
    "w/o": "without",
    "s/p": "status post",
    "h/o": "history of",
    "c/o": "complains of",
    "r/o": "rule out",
    "f/u": "follow up",
    "d/c": "discharge",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription",
    "sx": "symptoms",
    "hx": "history",
    "px": "physical examination",
    "abx": "antibiotics",
    "cp": "chest pain",
    "sob": "shortness of breath",
    "doe": "dyspnea on exertion",
    "n/v": "nausea and vomiting",
    "abd": "abdominal",
    "cv": "cardiovascular",
    "gi": "gastrointestinal",
    "gu": "genitourinary",
    "ms": "musculoskeletal",
    "neuro": "neurological",
    "psych": "psychiatric",
    "derm": "dermatological",
    "ent": "ear nose throat",
    "ob/gyn": "obstetrics and gynecology",
    "ed": "emergency department",
    "icu": "intensive care unit",
    "or": "operating room",
    "pacu": "post anesthesia care unit",
    "nicu": "neonatal intensive care unit",
    "picu": "pediatric intensive care unit",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "dm": "diabetes mellitus",
    "htn": "hypertension",
    "cad": "coronary artery disease",
    "pvd": "peripheral vascular disease",
    "esrd": "end stage renal disease",
    "ckd": "chronic kidney disease",
    "gerd": "gastroesophageal reflux disease",
    "uti": "urinary tract infection",
    "uri": "upper respiratory infection",
    "lri": "lower respiratory infection",
    "abg": "arterial blood gas",
    "cbc": "complete blood count",
    "bmp": "basic metabolic panel",
    "cmp": "comprehensive metabolic panel",
    "lfts": "liver function tests",
    "tsh": "thyroid stimulating hormone",
    "pt/inr": "prothrombin time/international normalized ratio",
    "ptt": "partial thromboplastin time",
    "ekg": "electrocardiogram",
    "ecg": "electrocardiogram",
    "cxr": "chest x-ray",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "us": "ultrasound",
    "echo": "echocardiogram",
    "eeg": "electroencephalogram",
    "emg": "electromyography",
    "pft": "pulmonary function test",
    "r": "right",
    "l": "left",
    "b/l": "bilateral",
    "prn": "as needed",
    "po": "by mouth",
    "iv": "intravenous",
    "im": "intramuscular",
    "sq": "subcutaneous",
    "sl": "sublingual",
    "pr": "per rectum",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qd": "daily",
    "qod": "every other day",
    "qhs": "at bedtime",
    "ac": "before meals",
    "pc": "after meals",
    "stat": "immediately",
    "asap": "as soon as possible",
    "npo": "nothing by mouth",
    "wbc": "white blood cell",
    "rbc": "red blood cell",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet",
    "na": "sodium",
    "k": "potassium",
    "cl": "chloride",
    "co2": "carbon dioxide",
    "bun": "blood urea nitrogen",
    "cr": "creatinine",
    "glu": "glucose",
    "ca": "calcium",
    "mg": "magnesium",
    "phos": "phosphorus",
    "tbili": "total bilirubin",
    "dbili": "direct bilirubin",
    "ast": "aspartate aminotransferase",
    "alt": "alanine aminotransferase",
    "alp": "alkaline phosphatase",
    "ggt": "gamma glutamyl transferase",
    "ldh": "lactate dehydrogenase",
    "ck": "creatine kinase",
    "troponin": "troponin",
    "bnp": "brain natriuretic peptide",
    "d-dimer": "d-dimer",
    "inr": "international normalized ratio",
    "aptt": "activated partial thromboplastin time",
    "fibrinogen": "fibrinogen",
    "esr": "erythrocyte sedimentation rate",
    "crp": "c-reactive protein",
    "rf": "rheumatoid factor",
    "ana": "antinuclear antibody",
    "hiv": "human immunodeficiency virus",
    "hbv": "hepatitis b virus",
    "hcv": "hepatitis c virus",
    "tb": "tuberculosis",
    "mrsa": "methicillin resistant staphylococcus aureus",
    "vre": "vancomycin resistant enterococcus",
    "cdiff": "clostridium difficile",
    "gcs": "glasgow coma scale",
    "loc": "level of consciousness",
    "a&o": "alert and oriented",
    "nad": "no acute distress",
    "nkda": "no known drug allergies",
    "pmh": "past medical history",
    "psh": "past surgical history",
    "fh": "family history",
    "sh": "social history",
    "ros": "review of systems",
    "vs": "vital signs",
    "hr": "heart rate",
    "bp": "blood pressure",
    "rr": "respiratory rate",
    "o2sat": "oxygen saturation",
    "temp": "temperature",
    "ht": "height",
    "wt": "weight",
    "bmi": "body mass index",
}

# Medical units
MEDICAL_UNITS = {
    "mg": "milligrams",
    "g": "grams",
    "kg": "kilograms",
    "mcg": "micrograms",
    "ml": "milliliters",
    "l": "liters",
    "dl": "deciliters",
    "meq": "milliequivalents",
    "mmol": "millimoles",
    "iu": "international units",
    "u": "units",
}


def expand_medical_abbreviations(text: str) -> str:
    """Expand common medical abbreviations."""
    # Convert to lowercase for matching
    text_lower = text.lower()

    # Sort abbreviations by length (longest first) to avoid partial replacements
    sorted_abbrevs = sorted(
        MEDICAL_ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True
    )

    for abbrev, expansion in sorted_abbrevs:
        # Use word boundaries to avoid partial matches
        pattern = r"\b" + re.escape(abbrev) + r"\b"
        text_lower = re.sub(pattern, expansion, text_lower, flags=re.IGNORECASE)

    return text_lower


def normalize_medical_numbers(text: str) -> str:
    """Normalize medical measurements and numbers."""
    # Normalize arrows and symbols
    text = text.replace("↑", " elevated")
    text = text.replace("↓", " decreased")
    text = text.replace("→", " indicates")
    text = text.replace("=", " equals")

    # Add spaces around numbers for better tokenization
    text = re.sub(r"(\d+)", r" \1 ", text)

    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_medical_text(text: str) -> str:
    """
    Preprocess medical text for better embedding quality.

    Args:
        text: Raw medical text

    Returns:
        Preprocessed text
    """
    # Expand abbreviations
    text = expand_medical_abbreviations(text)

    # Normalize numbers and symbols
    text = normalize_medical_numbers(text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def preprocess_batch(texts: List[str]) -> List[str]:
    """Preprocess a batch of medical texts."""
    return [preprocess_medical_text(text) for text in texts]
