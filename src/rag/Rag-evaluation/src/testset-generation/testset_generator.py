import os
import json
import getpass
import asyncio
import typing as t
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass

from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.llms.base import llm_factory
from ragas.testset.transforms import (
    apply_transforms,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    OverlapScoreBuilder,
)
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop import SingleHopQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

# --- Configuration ---
DOCS_DIR = "data/datasets/"
COMPONENTS_FILE = "data/components.json"
OUTPUT_DIR = "data/datasets"
OUTPUT_FILE = "testset.json"
NUM_SINGLE_HOP = 25
NUM_MULTI_HOP = 25
# --- End Configuration ---


@dataclass
class MySingleHopScenario(SingleHopQuerySynthesizer):
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):
        property_name = "keyphrases"
        nodes = [
            node
            for node in knowledge_graph.nodes
            if node.type.name == "CHUNK" and node.get_property(property_name)
        ]
        if not nodes:
            return []

        number_of_samples_per_node = max(1, n // len(nodes))
        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            themes = node.properties.get(property_name, [""])
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            base_scenarios = self.prepare_combinations(
                node,
                themes,
                personas=persona_list,
                persona_concepts=persona_concepts.mapping,
            )
            scenarios.extend(
                self.sample_combinations(base_scenarios, number_of_samples_per_node)
            )
        return scenarios


@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self, n: int, knowledge_graph, persona_list, callbacks
    ) -> t.List[MultiHopScenario]:
        results = knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: (
                True if rel.type == "keyphrases_overlap" else False
            )
        )
        if not results:
            return []

        num_sample_per_triplet = max(1, n // len(results))
        scenarios = []
        for triplet in results:
            if len(scenarios) >= n:
                break
            node_a, node_b = triplet[0], triplet[-1]
            overlapped_keywords = triplet[1].properties["overlapped_items"]
            if overlapped_keywords:
                themes = list(dict(overlapped_keywords).keys())
                prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
                persona_concepts = await self.theme_persona_matching_prompt.generate(
                    data=prompt_input, llm=self.llm, callbacks=callbacks
                )
                overlapped_keywords = [list(item) for item in overlapped_keywords]
                base_scenarios = self.prepare_combinations(
                    [node_a, node_b],
                    overlapped_keywords,
                    personas=persona_list,
                    persona_item_mapping=persona_concepts.mapping,
                    property_name="keyphrases",
                )
                base_scenarios = self.sample_diverse_combinations(
                    base_scenarios, num_sample_per_triplet
                )
                scenarios.extend(base_scenarios)
        return scenarios


async def main():
    """
    Generates a test set with single-hop and multi-hop questions.
    """
    # ----------------------------
    # 1. Load documents
    # ----------------------------
    md_loader = DirectoryLoader(DOCS_DIR, glob="**/*.md")
    md_docs = md_loader.load()

    try:
        with open(COMPONENTS_FILE, "r", encoding="utf-8") as f:
            components = json.load(f)
        json_docs = [
            Document(
                page_content=c.get("description", ""),
                metadata={"name": c.get("name"), "source": "components.json"},
            )
            for c in components
        ]
    except (FileNotFoundError, json.JSONDecodeError):
        json_docs = []

    all_docs = md_docs + json_docs
    if not all_docs:
        print("No documents found. Exiting.")
        return

    # ----------------------------
    # 2. Create Knowledge Graph
    # ----------------------------
    kg = KnowledgeGraph()
    for doc in all_docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    # ----------------------------
    # 3. Load environment and API keys
    # ----------------------------
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    # ----------------------------
    # 4. Set up LLM, Embeddings and Transforms
    # ----------------------------
    llm = llm_factory()

    headline_extractor = HeadlinesExtractor(llm=llm)
    headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
    keyphrase_extractor = KeyphrasesExtractor(
        llm=llm, property_name="keyphrases", max_num=10
    )
    relation_builder = OverlapScoreBuilder(
        property_name="keyphrases",
        new_property_name="overlap_score",
        threshold=0.01,
        distance_threshold=0.9,
    )
    transforms = [
        headline_extractor,
        headline_splitter,
        keyphrase_extractor,
        relation_builder,
    ]
    apply_transforms(kg, transforms=transforms)

    # ----------------------------
    # 5. Configure personas
    # ----------------------------
    persona1 = Persona(
        name="Maritime UI Developer",
        role_description="A frontend developer new to maritime applications, building user interfaces for ship bridge systems and needs to understand which components to use for different interaction patterns",
    )

    persona2 = Persona(
        name="UX Designer for Industrial Systems",
        role_description="A UX designer working on control interfaces for maritime and industrial equipment, focusing on creating intuitive and accessible user experiences",
    )

    persona3 = Persona(
        name="Maritime Software Architect",
        role_description="A senior developer designing the overall structure of maritime software applications and needs to understand component relationships and integration patterns",
    )

    persona4 = Persona(
        name="Junior Developer Learning OpenBridge",
        role_description="A junior developer who has been assigned to work with OpenBridge components but is unfamiliar with maritime interface design patterns and component usage",
    )

    persona5 = Persona(
        name="Product Manager for Maritime Software",
        role_description="A product manager overseeing maritime software development who needs to understand component capabilities to make informed decisions about feature requirements",
    )

    persona6 = Persona(
        name="QA Engineer for Maritime Interfaces",
        role_description="A quality assurance engineer testing maritime user interfaces and needs to understand expected component behaviors and interaction patterns",
    )

    persona7 = Persona(
        name="Accessibility Specialist",
        role_description="An accessibility expert ensuring maritime interfaces meet accessibility standards and needs to understand component accessibility features and proper usage",
    )

    persona8 = Persona(
        name="Integration Developer",
        role_description="A developer working on integrating OpenBridge components into existing maritime systems and needs detailed technical implementation guidance",
    )

    persona_list = [
        persona1,
        persona2,
        persona3,
        persona4,
        persona5,
        persona6,
        persona7,
        persona8,
    ]

    # ---------------------------
    # 6. Generate questions
    # ----------------------------
    single_hop_query_synth = MySingleHopScenario(llm=llm)
    multi_hop_query_synth = MyMultiHopQuery(llm=llm)

    print("Generating single-hop questions...")
    single_hop_scenarios = await single_hop_query_synth.generate_scenarios(
        n=NUM_SINGLE_HOP, knowledge_graph=kg, persona_list=persona_list
    )
    single_hop_results = [
        await single_hop_query_synth.generate_sample(s) for s in single_hop_scenarios
    ]

    print("Generating multi-hop questions...")
    multi_hop_scenarios = await multi_hop_query_synth.generate_scenarios(
        n=NUM_MULTI_HOP, knowledge_graph=kg, persona_list=persona_list
    )
    multi_hop_results = [
        await multi_hop_query_synth.generate_sample(s) for s in multi_hop_scenarios
    ]

    # ----------------------------
    # 7. Combine and Export
    # ----------------------------
    all_scenarios = single_hop_scenarios + multi_hop_scenarios
    all_results = single_hop_results + multi_hop_results
    test_data = []
    for scenario, result in zip(all_scenarios, all_results):
        contexts = [node.properties["page_content"] for node in scenario.nodes]
        metadata = []
        for node in scenario.nodes:
            if (
                node.type == NodeType.CHUNK
                and hasattr(node, "parent_id")
                and node.parent_id
            ):
                parent_node = kg.get_node(node.parent_id)
                if parent_node:
                    metadata.append(parent_node.properties.get("document_metadata", {}))
                else:
                    metadata.append({})
            elif "document_metadata" in node.properties:
                metadata.append(node.properties["document_metadata"])
            else:
                metadata.append({})
        test_data.append(
            {
                "question": result.user_input,
                "ground_truth": result.reference,
                "contexts": contexts,
                "metadata": metadata,
            }
        )

    dataset_df = pd.DataFrame(test_data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    dataset_df.to_json(output_path, orient="records", indent=4)

    print(f"✅ Testset generated and saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
