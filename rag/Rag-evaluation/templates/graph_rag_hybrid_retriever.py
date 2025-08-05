import os
from typing import List, Dict, Union
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.embeddings import SentenceTransformerEmbeddings
from llm_client import LocalLLMClient

# Neo4j imports - optional if not available
try:
    from langchain_neo4j import Neo4jGraph
    from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
# from ragbuilder.graph_utils.graph_loader import load_graph # This is a custom utility and not available


class GraphHybridRAG:
    def __init__(self, llm_client: LocalLLMClient):
        self.llm_client = llm_client
        self.embeddings = SentenceTransformerEmbeddings()

        # Initialize Neo4j graph if available and configured
        if NEO4J_AVAILABLE and all(
            [
                os.getenv("NEO4J_URI"),
                os.getenv("NEO4J_USER"),
                os.getenv("NEO4J_PASSWORD"),
            ]
        ):
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USER"),
                password=os.getenv("NEO4J_PASSWORD"),
            )
        else:
            self.graph = None

    def _generate_full_text_query(self, input_str: str) -> str:
        if not NEO4J_AVAILABLE:
            return ""
        words = [el for el in remove_lucene_chars(input_str).split() if el]
        if not words:
            return ""
        full_text_query = " AND ".join([f"{word}~2" for word in words])
        return full_text_query.strip()

    def _graph_retriever(self, question: str) -> str:
        if not self.graph:
            return "Graph database not available or not configured."

        try:

            class Entities(BaseModel):
                names: List[str] = Field(
                    ..., description="All nodes that appear in the text"
                )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are extracting entities from the text."),
                    (
                        "human",
                        "Use the given format to extract information from the following input: {question}",
                    ),
                ]
            )

            # Simple entity extraction fallback without structured output
            entity_response = self.llm_client.invoke(
                prompt.invoke({"question": question})
            )

            # Extract simple entity names from response
            entities_text = str(entity_response)
            entity_names = [
                word.strip() for word in entities_text.split() if len(word.strip()) > 2
            ][:5]

            if not entity_names:
                return ""

            result = ""
            for entity in entity_names:
                response = self.graph.query(
                    """
                    MATCH (e)
                    WHERE toLower(e.id) CONTAINS toLower($entity)
                    CALL {
                        WITH e
                        MATCH (e)-[r]->(n)
                        RETURN '`' + e.id + '` - ' + type(r) + ' -> `' + n.id + '`' AS output
                        UNION
                        WITH e
                        MATCH (e)<-[r]-(n)
                        RETURN '`' + n.id + '` - ' + type(r) + ' -> `' + e.id + '`' AS output
                    }
                    RETURN output
                    """,
                    {"entity": entity},
                )
                result += "\n".join([el["output"] for el in response])
            return result
        except Exception as e:
            return f"Error in graph retrieval: {e}"

    def run(
        self, question: str, reference_contexts: List[str]
    ) -> Dict[str, Union[str, List[str]]]:
        docs = [Document(page_content=content) for content in reference_contexts]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        documents = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embeddings
        )
        vector_retriever = vectorstore.as_retriever()

        # if os.getenv('NEO4J_LOAD', 'True').lower() == 'true':
        #     load_graph(documents, self.llm)

        def full_retriever(question_str: str):
            graph_data = self._graph_retriever(question_str)
            vector_data = [
                el.page_content for el in vector_retriever.invoke(question_str)
            ]
            return f"Graph data:\n{graph_data}\n\nVector data:\n{'#Document '.join(vector_data)}"

        # Get context and generate answer using classify_statement
        context_data = full_retriever(question)
        answer = self.llm_client.classify_statement(context_data, question)

        return {
            "answer": answer,
            "context": [context_data]
            if isinstance(context_data, str)
            else context_data,
        }
