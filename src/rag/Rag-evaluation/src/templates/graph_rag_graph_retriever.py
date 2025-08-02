import os
from typing import List, Dict, Union
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_neo4j import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from operator import itemgetter
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
# from ragbuilder.graph_utils.graph_loader import load_graph # This is a custom utility and not available


class GraphRAG:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

    def _generate_full_text_query(self, input_str: str) -> str:
        words = [el for el in remove_lucene_chars(input_str).split() if el]
        if not words:
            return ""
        full_text_query = " AND ".join([f"{word}~2" for word in words])
        return full_text_query.strip()

    def _graph_retriever(self, question: str) -> str:
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
        entity_chain = prompt | RunnableLambda(
            self.llm.with_structured_output(Entities).invoke
        )

        entities = entity_chain.invoke({"question": question})
        if not entities.names:
            return ""

        result = ""
        for entity in entities.names:
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

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            RunnableParallel(context=full_retriever, question=RunnablePassthrough())
            .assign(context=itemgetter("context"))
            .assign(answer=prompt | RunnableLambda(self.llm.invoke) | StrOutputParser())
            .pick(["answer", "context"])
        )

        result = rag_chain.invoke(question)

        if isinstance(result["context"], str):
            result["context"] = [result["context"]]

        return result
