from typing import List, Dict, Union
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate


class SemanticChunkerRAG:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n".join(doc.page_content for doc in docs)

    def run(
        self, question: str, reference_contexts: List[str]
    ) -> Dict[str, Union[str, List[str]]]:
        docs = [Document(page_content=content) for content in reference_contexts]

        text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )

        all_splits = []
        for doc in docs:
            splits = text_splitter.create_documents([doc.page_content])
            all_splits.extend(splits)

        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=self.embeddings
        )
        retriever = vectorstore.as_retriever()

        prompt_template_str = """
        You are a helpful AI assistant. Using the provided context, answer the user's question in a clear, factual manner (maximum 4 sentences).
        If the context does not contain sufficient information, simply reply that you don't know.

        <context>
        {context}
        </context>

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template_str.strip())

        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
            .assign(context=itemgetter("context") | RunnableLambda(self._format_docs))
            .assign(
                answer=prompt
                | RunnableLambda(self.llm.langchain_llm.invoke)
                | StrOutputParser()
            )
            .pick(["answer", "context"])
        )

        result = rag_chain.invoke(question)

        if isinstance(result["context"], str):
            result["context"] = [result["context"]]
        elif isinstance(result["context"], list) and all(
            isinstance(doc, Document) for doc in result["context"]
        ):
            result["context"] = [doc.page_content for doc in result["context"]]

        return result
