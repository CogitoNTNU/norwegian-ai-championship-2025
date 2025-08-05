"""
Enhanced Medical Ontology for Topic Classification
Comprehensive keyword mappings, synonyms, and medical term relationships
"""

from typing import Dict, List, Set, Tuple
import re

class MedicalOntology:
    """Enhanced medical knowledge base for topic classification."""
    
    def __init__(self):
        self.topic_keywords = self._build_comprehensive_mappings()
        self.medical_abbreviations = self._build_abbreviation_map()
        self.topic_hierarchy = self._build_topic_hierarchy()
        self.exclusion_patterns = self._build_exclusion_patterns()
    
    def _build_comprehensive_mappings(self) -> Dict[str, Dict[str, List[int]]]:
        """Build comprehensive medical keyword mappings with confidence scores."""
        return {
            # Cardiovascular System (High precision mappings)
            "cardiac": {
                "primary": [4, 7, 10, 11, 15, 22, 25, 38, 51, 49, 57, 77, 80, 82],  # ACS, MI, Aortic, Afib, etc.
                "secondary": [23, 76],  # Syncope (cardiac causes)
                "keywords": [
                    "heart", "cardiac", "cardio", "myocardial", "coronary", "aortic", "mitral", 
                    "tricuspid", "ventricular", "atrial", "pericardial", "endocardial",
                    "acs", "stemi", "nstemi", "mi", "infarction", "ischemia", "angina",
                    "arrhythmia", "tachycardia", "bradycardia", "fibrillation", "flutter",
                    "cardiomyopathy", "heart failure", "hf", "lvef", "ejection fraction",
                    "valve", "stenosis", "regurgitation", "murmur", "s3", "s4",
                    "chest pain", "crushing pain", "substernal", "precordial",
                    "ecg", "ekg", "troponin", "ck-mb", "bnp", "nt-probnp"
                ]
            },
            
            # Respiratory System
            "respiratory": {
                "primary": [8, 13, 14, 21, 45, 46, 47, 59, 60, 61, 62, 63, 64, 65, 66, 67, 74, 81],
                "secondary": [],
                "keywords": [
                    "lung", "pulmonary", "respiratory", "breath", "breathing", "dyspnea",
                    "pneumonia", "pneumothorax", "pleural", "bronchial", "alveolar",
                    "copd", "asthma", "ards", "respiratory failure", "hypoxia", "hypercapnia",
                    "wheeze", "stridor", "rales", "rhonchi", "crackles",
                    "spo2", "oxygen", "ventilation", "intubation", "mechanical ventilation",
                    "pe", "pulmonary embolism", "dvt", "pneumomediastinum",
                    "cough", "sputum", "hemoptysis", "chest x-ray", "ct chest"
                ]
            },
            
            # Neurological System
            "neurological": {
                "primary": [18, 29, 35, 48, 71, 75, 79],  # Brain death, delirium, encephalitis, etc.
                "secondary": [76],  # Syncope
                "keywords": [
                    "brain", "cerebral", "neurological", "neuro", "cns", "central nervous",
                    "stroke", "cva", "tia", "ischemic", "hemorrhagic", "subarachnoid",
                    "seizure", "epilepsy", "status epilepticus", "convulsion",
                    "meningitis", "encephalitis", "brain death", "coma", "gcs",
                    "delirium", "confusion", "altered mental status", "ams",
                    "headache", "migraine", "tension headache",
                    "hemiplegia", "hemiparesis", "aphasia", "dysarthria",
                    "lumbar puncture", "csf", "ct head", "mri brain"
                ]
            },
            
            # Gastrointestinal System
            "gastrointestinal": {
                "primary": [1, 2, 3, 6, 17, 37, 54, 56],  # Acute abdomen, appendicitis, etc.
                "secondary": [27],  # Non-cardiac chest pain (GERD)
                "keywords": [
                    "abdomen", "abdominal", "stomach", "gastric", "intestinal", "bowel",
                    "gi", "gastrointestinal", "digestive", "hepatic", "liver",
                    "appendicitis", "cholecystitis", "pancreatitis", "diverticulitis",
                    "obstruction", "ileus", "perforation", "peritonitis",
                    "nausea", "vomiting", "diarrhea", "constipation", "melena", "hematochezia",
                    "gi bleeding", "upper gi", "lower gi", "hematemesis",
                    "abdominal pain", "mcburney", "murphy", "rebound tenderness",
                    "lipase", "amylase", "alt", "ast", "bilirubin", "alkaline phosphatase"
                ]
            },
            
            # Trauma and Emergency
            "trauma": {
                "primary": [0, 16, 20, 23, 26, 28, 39, 55, 68, 70, 79],  # Various trauma types
                "secondary": [],
                "keywords": [
                    "trauma", "injury", "accident", "fall", "mvc", "motor vehicle",
                    "blunt", "penetrating", "laceration", "contusion", "fracture",
                    "head trauma", "tbi", "traumatic brain injury", "concussion",
                    "spinal", "cervical spine", "c-spine", "spinal cord",
                    "chest trauma", "hemothorax", "pneumothorax", "flail chest",
                    "abdominal trauma", "hemoperitoneum", "splenic laceration",
                    "burns", "thermal", "chemical", "electrical", "inhalation injury",
                    "fast exam", "trauma bay", "primary survey", "secondary survey"
                ]
            },
            
            # Infectious Diseases
            "infectious": {
                "primary": [9, 35, 36, 48, 72, 87, 89, 104],  # Sepsis, meningitis, etc.
                "secondary": [61],  # Pneumonia
                "keywords": [
                    "sepsis", "septic", "infection", "infectious", "bacteremia",
                    "fever", "febrile", "hyperthermia", "temperature", "chills", "rigors",
                    "antibiotic", "antimicrobial", "resistance", "mrsa", "vre",
                    "blood culture", "culture", "sensitivity", "gram stain",
                    "white blood cell", "wbc", "leukocytosis", "leukopenia",
                    "procalcitonin", "crp", "c-reactive protein", "esr",
                    "meningitis", "encephalitis", "endocarditis", "cellulitis"
                ]
            },
            
            # Metabolic and Endocrine
            "metabolic": {
                "primary": [30, 31, 42, 43, 44, 50, 52, 53, 58, 73],  # DKA, electrolyte disorders
                "secondary": [],
                "keywords": [
                    "diabetes", "diabetic", "dka", "ketoacidosis", "hyperglycemia", "hypoglycemia",
                    "insulin", "glucose", "blood sugar", "a1c", "hemoglobin a1c",
                    "electrolyte", "sodium", "potassium", "chloride", "co2", "anion gap",
                    "hyponatremia", "hypernatremia", "hypokalemia", "hyperkalemia",
                    "dehydration", "fluid", "crystalloid", "normal saline", "lactated ringers",
                    "thyroid", "tsh", "hyperthyroid", "hypothyroid", "thyrotoxicosis"
                ]
            },
            
            # Diagnostic Tests and Procedures
            "diagnostic": {
                "primary": [83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                "secondary": [],
                "keywords": [
                    "ecg", "ekg", "electrocardiogram", "12-lead", "rhythm strip",
                    "chest x-ray", "cxr", "ct", "mri", "ultrasound", "echo", "echocardiogram",
                    "lab", "laboratory", "blood work", "cbc", "bmp", "cmp", "coagulation",
                    "pt", "ptt", "inr", "d-dimer", "fibrinogen",
                    "arterial blood gas", "abg", "ph", "pco2", "po2", "bicarbonate",
                    "urinalysis", "ua", "urine", "microscopy", "dipstick",
                    "lumbar puncture", "lp", "csf", "cerebrospinal fluid",
                    "stress test", "exercise", "pharmacologic", "dobutamine"
                ]
            }
        }
    
    def _build_abbreviation_map(self) -> Dict[str, str]:
        """Map medical abbreviations to full terms."""
        return {
            # Cardiovascular
            "mi": "myocardial infarction", "acs": "acute coronary syndrome",
            "stemi": "st elevation myocardial infarction", "nstemi": "non st elevation myocardial infarction",
            "hf": "heart failure", "chf": "congestive heart failure", "lvef": "left ventricular ejection fraction",
            "afib": "atrial fibrillation", "aflutter": "atrial flutter", "vt": "ventricular tachycardia",
            "vf": "ventricular fibrillation", "svt": "supraventricular tachycardia",
            
            # Respiratory
            "copd": "chronic obstructive pulmonary disease", "ards": "acute respiratory distress syndrome",
            "pe": "pulmonary embolism", "dvt": "deep vein thrombosis", "sob": "shortness of breath",
            
            # Neurological
            "cva": "cerebrovascular accident", "tia": "transient ischemic attack",
            "gcs": "glasgow coma scale", "ams": "altered mental status", "cns": "central nervous system",
            
            # Gastrointestinal
            "gi": "gastrointestinal", "gerd": "gastroesophageal reflux disease",
            "pud": "peptic ulcer disease", "ibd": "inflammatory bowel disease",
            
            # Laboratory
            "cbc": "complete blood count", "bmp": "basic metabolic panel", "cmp": "comprehensive metabolic panel",
            "pt": "prothrombin time", "ptt": "partial thromboplastin time", "inr": "international normalized ratio",
            "abg": "arterial blood gas", "ua": "urinalysis", "csf": "cerebrospinal fluid",
            
            # Imaging
            "cxr": "chest x-ray", "ct": "computed tomography", "mri": "magnetic resonance imaging",
            "echo": "echocardiogram", "ekg": "electrocardiogram", "ecg": "electrocardiogram",
            
            # Emergency/Trauma
            "mvc": "motor vehicle collision", "tbi": "traumatic brain injury", "c-spine": "cervical spine",
            "fast": "focused assessment with sonography for trauma", "ed": "emergency department"
        }
    
    def _build_topic_hierarchy(self) -> Dict[int, Dict[str, any]]:
        """Build hierarchical relationships between topics."""
        return {
            # Cardiac hierarchy
            4: {"parent": None, "children": [7], "weight": 1.0, "category": "cardiac"},      # ACS
            7: {"parent": 4, "children": [], "weight": 1.0, "category": "cardiac"},         # MI
            10: {"parent": None, "children": [], "weight": 1.0, "category": "cardiac"},     # Aortic dissection
            25: {"parent": None, "children": [77], "weight": 1.0, "category": "cardiac"},   # Cardiomyopathy
            77: {"parent": 25, "children": [], "weight": 0.9, "category": "cardiac"},       # Takotsubo
            
            # Respiratory hierarchy
            61: {"parent": None, "children": [13], "weight": 1.0, "category": "respiratory"}, # Pneumonia
            13: {"parent": 61, "children": [], "weight": 0.8, "category": "respiratory"},     # Aspiration pneumonia
            63: {"parent": None, "children": [], "weight": 1.0, "category": "respiratory"},   # PE
            
            # GI hierarchy
            1: {"parent": None, "children": [2, 3], "weight": 1.0, "category": "gi"},      # Acute abdomen
            2: {"parent": 1, "children": [], "weight": 0.9, "category": "gi"},             # Appendicitis
            3: {"parent": 1, "children": [], "weight": 0.9, "category": "gi"},             # Cholecystitis
        }
    
    def _build_exclusion_patterns(self) -> Dict[str, List[str]]:
        """Build patterns that exclude certain topics."""
        return {
            "cardiac_exclusions": ["dental", "tooth", "orthodontic", "gum"],
            "respiratory_exclusions": ["cardiac arrest", "heart failure"],
            "gi_exclusions": ["chest pain", "myocardial"],
        }
    
    def expand_text(self, text: str) -> str:
        """Expand abbreviations in text."""
        expanded = text.lower()
        for abbrev, full_term in self.medical_abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded = re.sub(pattern, full_term, expanded)
        return expanded
    
    def get_topic_candidates(self, text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Get ranked topic candidates with confidence scores."""
        expanded_text = self.expand_text(text)
        topic_scores = {}
        
        for category, info in self.topic_keywords.items():
            primary_topics = info["primary"]
            secondary_topics = info.get("secondary", [])
            keywords = info["keywords"]
            
            # Calculate keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in expanded_text)
            if keyword_matches == 0:
                continue
            
            # Score calculation with keyword density
            keyword_density = keyword_matches / len(keywords)
            text_length_factor = min(1.0, len(expanded_text) / 100)  # Normalize by text length
            
            # Primary topics get higher scores
            for topic in primary_topics:
                base_score = keyword_density * 1.0 * text_length_factor
                hierarchy_info = self.topic_hierarchy.get(topic, {})
                weight = hierarchy_info.get("weight", 1.0)
                topic_scores[topic] = base_score * weight
            
            # Secondary topics get lower scores
            for topic in secondary_topics:
                base_score = keyword_density * 0.7 * text_length_factor
                hierarchy_info = self.topic_hierarchy.get(topic, {})
                weight = hierarchy_info.get("weight", 1.0)
                topic_scores[topic] = base_score * weight * 0.8
        
        # Sort by score and return top candidates
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:top_k]
    
    def get_focused_topic_list(self, text: str, max_topics: int = 15) -> List[int]:
        """Get a focused list of most relevant topics for classification."""
        candidates = self.get_topic_candidates(text, top_k=max_topics)
        
        if not candidates:
            # Default emergency topics if no matches
            return [1, 7, 16, 25, 27, 37, 61, 72, 75, 76, 83]
        
        # Return just the topic IDs
        return [topic_id for topic_id, score in candidates]
