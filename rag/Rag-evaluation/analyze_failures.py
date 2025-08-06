from templates.biobert_diagnostic import BiobertDiagnostic
from llm_client import LocalLLMClient


def analyze_failures():
    """Provides a detailed sentence-level analysis of failed retrieval cases."""
    # Initialize systems
    llm_client = LocalLLMClient(model_name="cogito:14b")
    diagnostic = BiobertDiagnostic(llm_client=llm_client)

    # --- Case 1: MRI breast cancer screening ---
    print("=" * 80)
    print("DEEP DIVE: Case 1 - MRI breast cancer screening (Ground Truth: False)")
    print("=" * 80)
    question1 = "Breast cancer screening with MRI is indicated for patients with greater than 15% lifetime risk of breast cancer, BRCA gene carriers, and those with a history of chest radiation exposure."
    print(f"Question: {question1}")
    print(
        "Ground Truth Analysis: The statement is FALSE because the actual recommended threshold is >20-25%, not >15%."
    )

    results1 = diagnostic.run_all_retrievers(question1)

    print("\n🔍 SEMANTIC SEARCH (BioBERT) ANALYSIS:")
    for i, result in enumerate(results1["semantic_results"]):
        doc = result["document"]
        text = doc["text"]
        print(
            f"  Result {i + 1} | Score: {result['score']:.2f} | Topic: {doc.get('topic_name')}"
        )
        print(f"    -> Text Snippet: {text[:250]}...")
        # Check for the specific factual claim
        contains_15 = "15%" in text
        contains_20 = "20%" in text
        contains_screening = "breast cancer screening" in text.lower()
        print(
            f"    -> Analysis: Contains >15%? {contains_15} | Contains >20%? {contains_20} | Mentions screening? {contains_screening}"
        )
        if not contains_15 and not contains_20:
            print(
                "    -> 🚨 FAILURE: Retrieved document is about MRI but lacks the specific risk percentage."
            )
        print()

    print("\n🔑 TOPIC-CLASSIFIED BM25 ANALYSIS:")
    for i, result in enumerate(results1["topic_classified_bm25_results"]):
        doc = result["document"]
        text = doc["text"]
        print(
            f"  Result {i + 1} | Score: {result['score']:.2f} | Topic: {doc.get('topic_name')}"
        )
        print(f"    -> Text Snippet: {text[:250]}...")
        # Check for the specific factual claim
        contains_15 = "15%" in text
        contains_20 = "20%" in text
        contains_screening = "breast cancer screening" in text.lower()
        print(
            f"    -> Analysis: Contains >15%? {contains_15} | Contains >20%? {contains_20} | Mentions screening? {contains_screening}"
        )
        print()

    # --- Case 2: Takotsubo Cardiomyopathy ---
    print("\n" + "=" * 80)
    print("DEEP DIVE: Case 2 - Takotsubo Cardiomyopathy (Ground Truth: False)")
    print("=" * 80)
    question2 = "The histopathologic findings in mid-ventricular Takotsubo cardiomyopathy consistently show evidence of myocardial necrosis and diffuse lymphocytic infiltration."
    print(f"Question: {question2}")
    print(
        "Ground Truth Analysis: The statement is FALSE. Takotsubo is characterized by the ABSENCE of necrosis and inflammation."
    )

    results2 = diagnostic.run_all_retrievers(question2)

    print("\n🔑 TOPIC-CLASSIFIED BM25 ANALYSIS (BEST PERFORMING METHOD):")
    topic_bm25_results = results2["topic_classified_bm25_results"]
    classified_topic = (
        topic_bm25_results[0].get("classified_topic", "Unknown")
        if topic_bm25_results
        else "Unknown"
    )
    print(f"  -> Classified Topic: {classified_topic}")

    for i, result in enumerate(topic_bm25_results):
        doc = result["document"]
        text = doc["text"]
        print(
            f"  Result {i + 1} | Score: {result['score']:.2f} | Topic: {doc.get('topic_name')}"
        )
        print(f"    -> Text Snippet: {text[:250]}...")
        # Check for the specific factual claim
        contains_necrosis = "necrosis" in text.lower()
        contains_absence = (
            "absence of necrosis" in text.lower() or "without necrosis" in text.lower()
        )
        print(
            f"    -> Analysis: Mentions necrosis? {contains_necrosis} | Mentions ABSENCE of necrosis? {contains_absence}"
        )
        if not contains_absence:
            print(
                "    -> 🚨 FAILURE: Retrieved document discusses the condition but does not contain the crucial contradictory evidence (the absence of necrosis)."
            )
        print()


if __name__ == "__main__":
    analyze_failures()
