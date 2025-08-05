from typing import List, Dict, Union
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain import hub
from langchain.load import dumps, loads
from langchain_community.embeddings import SentenceTransformerEmbeddings
from llm_client import LocalLLMClient


class QueryExpansionRRF:
    def __init__(self, llm_client: LocalLLMClient):
        self.llm_client = llm_client
        self.embeddings = SentenceTransformerEmbeddings()

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n".join(doc.page_content for doc in docs)

    def _rrf(self, results: List[List[Document]]) -> List[Document]:
        fused_scores = {}
        k = 60
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            loads(doc)
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return reranked_results

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

        template = """You are a helpful assistant that generates multiple search queries based on a single input query.
        Generate multiple search queries related to: {question}
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion
            | RunnableLambda(self.llm_client.invoke)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        retrieval_chain_rag_fusion = generate_queries | retriever.map()
        results = retrieval_chain_rag_fusion.invoke({"question": question})

        reranked_docs = self._rrf(results)

        context = self._format_docs(reranked_docs)

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = prompt | RunnableLambda(self.llm_client.invoke) | StrOutputParser()

        # Format the answer using classify_statement
        answer = self.llm_client.classify_statement(context, question)

        return {
            "answer": answer,
            "context": [doc.page_content for doc in reranked_docs],
        }
