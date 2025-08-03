import time
from typing import List, Dict, Union
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from operator import itemgetter


class ContextualRetrieverRAG:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.rag_chain = self._create_rag_chain()

    def _create_rag_chain(self):
        # This is a placeholder. The actual chain will be created dynamically in the run method
        # because it depends on the reference_contexts.
        return None

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n".join(doc.page_content for doc in docs)

    def run(
        self, question: str, reference_contexts: List[str]
    ) -> Dict[str, Union[str, List[str]]]:
        docs = [Document(page_content=content) for content in reference_contexts]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        document_array = []
        for i, split in enumerate(splits):
            chunk_group = splits[max(0, i - 5) : min(len(splits), i + 5)]
            chunk_content = self._format_docs([split])
            chunk_group_content = self._format_docs(chunk_group)

            prompt_template = f"""
            <document> 
             {chunk_group_content}
            </document> 
            Here is the chunk we want to situate within the chunk group 
            <chunk> 
            {chunk_content}
            </chunk>
            {{question}}
            """
            prompt_template = prompt_template.replace("{{", "{").replace("}}", "}")
            cr_prompt = ChatPromptTemplate.from_template(prompt_template)
            cr_prompt_chain = cr_prompt | self.llm.langchain_llm | StrOutputParser()

            contextualized_content = cr_prompt_chain.invoke(
                {
                    "question": "Please give a short succinct context to situate this chunk within the overall document (the group of 10 chunks) for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
                }
            )

            document = Document(
                page_content=contextualized_content + " " + chunk_content,
                metadata={"id": i},
            )
            document_array.append(document)

        timestamp = str(int(time.time()))
        index_name = "contextual-retriever-" + timestamp

        vectorstore = Chroma.from_documents(
            documents=document_array,
            embedding=self.embeddings,
            collection_name=index_name,
        )

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )
        bm25_retriever = BM25Retriever.from_documents(splits)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
        )

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
            RunnableParallel(context=ensemble_retriever, question=RunnablePassthrough())
            .assign(context=itemgetter("context") | RunnableLambda(self._format_docs))
            .assign(answer=prompt | self.llm.langchain_llm | StrOutputParser())
            .pick(["answer", "context"])
        )

        result = rag_chain.invoke(question)

        # Ensure context is a list of strings
        if isinstance(result["context"], str):
            result["context"] = [result["context"]]
        elif isinstance(result["context"], list) and all(
            isinstance(doc, Document) for doc in result["context"]
        ):
            result["context"] = [doc.page_content for doc in result["context"]]

        return result
