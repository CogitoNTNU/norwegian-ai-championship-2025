from typing import List, Dict, Union
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import SentenceTransformerEmbeddings
from llm_client import LocalLLMClient


class StepBackPromptingRAG:
    def __init__(self, llm_client: LocalLLMClient):
        self.llm_client = llm_client
        self.embeddings = SentenceTransformerEmbeddings()

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n".join(doc.page_content for doc in docs)

    def run(
        self, question: str, reference_contexts: List[str]
    ) -> Dict[str, Union[str, List[str]]]:
        docs = [Document(page_content=content) for content in reference_contexts]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings)
        retriever = vectorstore.as_retriever()

        # Error handling with fallback
        if not self.llm_client:
            raise ValueError("LLM Client must be initialized.")

        rewrite_prompt_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
        Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
        Original query: {question}"""
        query_rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_template)

        query_rewrite_chain = (
            query_rewrite_prompt
            | RunnableLambda(self.llm_client.invoke)
            | StrOutputParser()
        )

        step_back_question = query_rewrite_chain.invoke({"question": question})

        retrieved_docs = retriever.invoke(step_back_question)
        context = self._format_docs(retrieved_docs)

        prompt_template_str = """
        You are a helpful AI assistant. Using the provided context, answer the user's question in a clear, factual manner (maximum 4 sentences).
        If the context does not contain sufficient information, simply reply that you don't know.

        <context>
        {context}
        </context>

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template_str.strip())

        # Format the answer using classify_statement
        answer = self.llm_client.classify_statement(context, question)

        return {
            "answer": answer,
            "context": [doc.page_content for doc in retrieved_docs],
        }
