from langchain_community.document_loaders import *
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

class HybridRAG:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def run(self, question, context):
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs)

        docs = []
        for i, c in enumerate(context):
            with open(f"temp_context_{i}.txt", "w") as f:
                f.write(c)
            loader = TextLoader(f"temp_context_{i}.txt")
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        c = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="testindex-ragbuilder",
        )
        retrievers = []
        retriever = c.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrievers.append(retriever)
        bm25_retriever = BM25Retriever.from_documents(splits)
        retrievers.append(bm25_retriever)
        retriever = MergerRetriever(retrievers=retrievers)
        # Define RAG prompt locally instead of using hub
        prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
        prompt = PromptTemplate.from_template(prompt_template)
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
            .assign(context=itemgetter("context") | RunnableLambda(format_docs))
            .assign(answer=prompt | self.llm.langchain_llm | StrOutputParser())
            .pick(["answer", "context"])
        )
        return rag_chain.invoke(question)
