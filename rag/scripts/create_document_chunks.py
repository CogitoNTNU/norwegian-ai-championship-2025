#!/usr/bin/env python3
"""
Phase 2: Document Chunking System for Rich Context RAG
Creates 400-word chunks from comprehensive medical documents with topic metadata.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


class DocumentChunker:
    """Creates rich document chunks with topic metadata for Phase 2 RAG system."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.topic_mapping = self._create_topic_mapping()

    def _create_topic_mapping(self) -> Dict[str, int]:
        """Create topic mapping based on directory structure."""
        topics = {
            "12-lead ECG": 1,
            "Abdominal Trauma": 2,
            "Acute Abdomen": 3,
            "Acute Appendicitis": 4,
            "Acute Cholecystitis": 5,
            "Acute Coronary Syndrome": 6,
            "Acute Kidney Injury": 7,
            "Acute Liver Failure": 8,
            "Acute Myocardial Infarction (STEMI_NSTEMI)": 9,
            "Acute Respiratory Distress Syndrome": 10,
            "Anaphylaxis": 11,
            "Angiography (invasive)": 12,
            "Aortic Dissection": 13,
            "Aortic Stenosis": 14,
            "Arrhythmias (various)": 15,
            "Arterial Blood Gas (ABG)": 16,
            "Aspiration Pneumonia": 17,
            "Asthma Exacerbation": 18,
            "Atrial Fibrillation": 19,
            "B-type Natriuretic Peptide (BNP)": 20,
            "Blood Cultures": 21,
            "Blunt Trauma": 22,
            "Bowel Obstruction": 23,
            "Brain Death": 24,
            "Bronchitis": 25,
            "Bronchoscopy with BAL": 26,
            "Burns": 27,
            "C-Reactive Protein (CRP)": 28,
            "COPD Exacerbation": 29,
            "CT Angiogram": 30,
            "CT Other": 31,
            "Cardiac Arrest": 32,
            "Cardiac Catheterization": 33,
            "Cardiac Contusion": 34,
            "Cardiac Tamponade": 35,
            "Cardiomyopathy": 36,
            "Central Venous Pressure": 37,
            "Cervical Spine Injury": 38,
            "Chest Pain (non-cardiac)": 39,
            "Chest X-ray": 40,
            "Coagulation Studies (PT_PTT_INR)": 41,
            "Compartment Syndrome": 42,
            "Complete Blood Count (CBC) with differential": 43,
            "Creatine Phosphokinase": 44,
            "Delirium": 45,
            "Diabetic Ketoacidosis": 46,
            "Echocardiogram": 47,
            "Eclampsia": 48,
            "Ectopic Pregnancy": 49,
            "Embolism": 50,
            "Empyema": 51,
            "Encephalitis": 52,
            "Endocarditis": 53,
            "GI Bleeding": 54,
            "Heart Failure (Acute_Chronic)": 55,
            "Hemoglobin A1C": 56,
            "Hemothorax": 57,
            "Hypertensive Emergency": 58,
            "Hyperventilation Syndrome": 59,
            "Hypoglycemia": 60,
            "Hyponatremia_Hypernatremia": 61,
            "Hypothermia_Hyperthermia": 62,
            "Interstitial Lung Disease": 63,
            "Lactate": 64,
            "Lipase": 65,
            "Lumbar Puncture_CSF Analysis": 66,
            "Lung Abscess": 67,
            "Lung Cancer": 68,
            "MRI": 69,
            "Meningitis": 70,
            "Mitral Regurgitation": 71,
            "Multi-organ Failure": 72,
            "Myocarditis": 73,
            "Ovarian Torsion": 74,
            "Overdose_Poisoning": 75,
            "Pancreatitis": 76,
            "Penetrating Trauma": 77,
            "Perforated Viscus": 78,
            "Pericarditis": 79,
            "Placental Abruption": 80,
            "Pleural Effusion": 81,
            "Pneumomediastinum": 82,
            "Pneumonia (bacterial_viral_atypical)": 83,
            "Pneumothorax": 84,
            "Procalcitonin": 85,
            "Pulmonary Embolism": 86,
            "Pulmonary Fibrosis": 87,
            "Pulmonary Hypertension": 88,
            "Pulse Oximetry_Arterial Saturation": 89,
            "Respiratory Acidosis": 90,
            "Respiratory Failure": 91,
            "Rhabdomyolysis": 92,
            "Right Heart Failure": 93,
            "Ruptured AAA": 94,
            "Seizures_Status Epilepticus": 95,
            "Sepsis_Septic Shock": 96,
            "Shock (various types)": 97,
            "Sleep Apnea": 98,
            "Stress Test (exercise or pharmacologic)": 99,
            "Stroke (Ischemic_Hemorrhagic)": 100,
            "Syncope": 101,
            "Takotsubo Cardiomyopathy": 102,
            "Testicular Torsion": 103,
            "Thyroid Stimulating Hormone (TSH)": 104,
            "Toxicology Screen": 105,
            "Traumatic Brain Injury": 106,
            "Troponin I_T": 107,
            "Ultrasound Doppler (extremities)": 108,
            "Ultrasound FAST exam": 109,
            "Ultrasound Other": 110,
            "Unstable Angina": 111,
            "Upper Airway Obstruction": 112,
            "Urinalysis": 113,
            "Urine Pregnancy Test": 114,
            "Ventricular Tachycardia": 115,
        }
        return topics

    def _extract_topic_from_path(self, file_path: Path) -> Tuple[str, int]:
        """Extract topic name and ID from file path."""
        # Get the parent directory name which contains the topic
        topic_dir = file_path.parent.name

        # Find matching topic
        for topic_name, topic_id in self.topic_mapping.items():
            if topic_dir == topic_name:
                return topic_name, topic_id

        # Fallback: use directory name as topic
        return topic_dir, 999  # Unknown topic ID

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove multiple whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove extra newlines but preserve paragraph structure
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def _create_chunk(
        self,
        words: List[str],
        chunk_idx: int,
        file_path: Path,
        topic_name: str,
        topic_id: int,
    ) -> Dict:
        """Create a single document chunk with metadata."""
        chunk_text = " ".join(words)

        # Create unique chunk ID
        doc_name = file_path.stem
        chunk_id = f"{doc_name}_chunk_{chunk_idx}"

        return {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "source_document": doc_name,
            "topic_name": topic_name,
            "topic_id": topic_id,
            "chunk_index": chunk_idx,
            "word_count": len(words),
            "source_path": str(file_path),
        }

    def chunk_document(self, file_path: Path) -> List[Dict]:
        """Create overlapping chunks from a single document."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        # Clean and prepare text
        content = self._clean_text(content)
        words = content.split()

        if len(words) < 50:  # Skip very short documents
            return []

        # Extract topic information
        topic_name, topic_id = self._extract_topic_from_path(file_path)

        chunks = []
        chunk_idx = 0

        # Create overlapping chunks
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]

            chunk = self._create_chunk(
                chunk_words, chunk_idx, file_path, topic_name, topic_id
            )
            chunks.append(chunk)

            chunk_idx += 1

            # Move start position with overlap
            if end_idx >= len(words):
                break
            start_idx = end_idx - self.overlap

        return chunks

    def process_all_documents(self, input_dir: Path, output_dir: Path) -> None:
        """Process all markdown documents and create chunks."""
        output_dir.mkdir(parents=True, exist_ok=True)

        all_chunks = []
        chunk_metadata = []

        print(f"Processing documents from: {input_dir}")

        # Process all .md files
        md_files = list(input_dir.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files")

        for file_path in md_files:
            print(f"Processing: {file_path.relative_to(input_dir)}")
            chunks = self.chunk_document(file_path)

            for chunk in chunks:
                # Save individual chunk file
                chunk_file_path = output_dir / f"{chunk['chunk_id']}.txt"
                with open(chunk_file_path, "w", encoding="utf-8") as f:
                    f.write(chunk["text"])

                # Store metadata
                chunk_metadata.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "source_document": chunk["source_document"],
                        "topic_name": chunk["topic_name"],
                        "topic_id": chunk["topic_id"],
                        "chunk_index": chunk["chunk_index"],
                        "word_count": chunk["word_count"],
                        "file_path": str(chunk_file_path),
                    }
                )

                all_chunks.append(chunk)

        # Save comprehensive metadata
        metadata_file = output_dir / "chunk_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chunk_metadata, f, indent=2)

        # Save topic mapping
        topic_mapping_file = output_dir / "topic_mapping.json"
        with open(topic_mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.topic_mapping, f, indent=2)

        print("\nProcessing complete!")
        print(f"Created {len(all_chunks)} document chunks")
        print(f"Metadata saved to: {metadata_file}")
        print(f"Topic mapping saved to: {topic_mapping_file}")


def main():
    """Main function to run the document chunking process."""
    # Paths
    input_dir = Path("data/raw/topics")
    output_dir = Path("data/processed/document_chunks")

    # Ensure input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    # Create chunker and process documents
    chunker = DocumentChunker(chunk_size=400, overlap=50)
    chunker.process_all_documents(input_dir, output_dir)


if __name__ == "__main__":
    main()
