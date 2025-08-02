"""Medical-specific embedding models with preprocessing."""

import re
from typing import List, Union
import numpy as np

from .sentence_transformers import MedicalEmbeddingModel


def medical_text_preprocessor(text: str) -> str:
    """
    Preprocess medical text for better embeddings.

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    # Expand common medical abbreviations
    abbreviations = {
        r"\bECG\b": "electrocardiogram",
        r"\bEKG\b": "electrocardiogram",
        r"\bMI\b": "myocardial infarction",
        r"\bCT\b": "computed tomography",
        r"\bMRI\b": "magnetic resonance imaging",
        r"\bICU\b": "intensive care unit",
        r"\bER\b": "emergency room",
        r"\bED\b": "emergency department",
        r"\bBP\b": "blood pressure",
        r"\bHR\b": "heart rate",
        r"\bRR\b": "respiratory rate",
        r"\bO2\b": "oxygen",
        r"\bCOPD\b": "chronic obstructive pulmonary disease",
        r"\bUTI\b": "urinary tract infection",
        r"\bDVT\b": "deep vein thrombosis",
        r"\bPE\b": "pulmonary embolism",
        r"\bAFib\b": "atrial fibrillation",
        r"\bCHF\b": "congestive heart failure",
        r"\bDKA\b": "diabetic ketoacidosis",
        r"\bGI\b": "gastrointestinal",
        r"\bCBC\b": "complete blood count",
        r"\bWBC\b": "white blood cell",
        r"\bRBC\b": "red blood cell",
    }

    # Apply abbreviation expansion
    processed = text
    for abbrev, expansion in abbreviations.items():
        processed = re.sub(abbrev, expansion, processed, flags=re.IGNORECASE)

    # Normalize medical units
    units = {
        r"(\d+)\s*mmHg": r"\1 millimeters of mercury",
        r"(\d+)\s*mg/dL": r"\1 milligrams per deciliter",
        r"(\d+)\s*mL/min": r"\1 milliliters per minute",
        r"(\d+)\s*bpm": r"\1 beats per minute",
    }

    for pattern, replacement in units.items():
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

    # Clean up extra spaces
    processed = " ".join(processed.split())

    return processed


class PubMedBERTEmbeddings(MedicalEmbeddingModel):
    """PubMedBERT embeddings with medical preprocessing."""

    def __init__(self, **kwargs):
        """Initialize PubMedBERT embeddings."""
        super().__init__(
            model_name="NeuML/pubmedbert-base-embeddings",
            preprocessing_fn=medical_text_preprocessor,
            **kwargs,
        )


class BioLORDEmbeddings(MedicalEmbeddingModel):
    """BioLORD embeddings for biomedical text."""

    def __init__(self, **kwargs):
        """Initialize BioLORD embeddings."""
        super().__init__(
            model_name="FremyCompany/BioLORD-2023",
            preprocessing_fn=medical_text_preprocessor,
            **kwargs,
        )


class MedCPTEmbeddings(MedicalEmbeddingModel):
    """MedCPT embeddings for medical information retrieval."""

    def __init__(self, **kwargs):
        """Initialize MedCPT embeddings."""
        # Note: MedCPT requires special handling for query vs document encoding
        super().__init__(model_name="ncbi/MedCPT-Query-Encoder", **kwargs)

        # For document encoding, we'd use ncbi/MedCPT-Article-Encoder
        self.is_query_encoder = True

    def encode_documents(
        self, documents: Union[str, List[str]], **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode documents with document encoder.

        Args:
            documents: Documents to encode
            **kwargs: Additional parameters

        Returns:
            Document embeddings
        """
        # This would use a separate document encoder
        # For now, we use the same encoder
        return self.encode(documents, **kwargs)
