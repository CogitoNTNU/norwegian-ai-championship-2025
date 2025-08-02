from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class HyDE:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def run(self, question, context):
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs)

        # Generate hypothetical answer
        pre_prompt = """You are an AI expert in the topic of the user question.
        I'm going to ask you a question, please generate a hypothetical answer.
        This answer will be used to fetch documents related to the question.
        Question: """
        hyde_prompt = PromptTemplate.from_template(pre_prompt + "{question}")
        hypothetical_answer_chain = (
            hyde_prompt | self.llm.langchain_llm | StrOutputParser()
        )

        # Get the hypothetical answer
        hypothetical_answer = hypothetical_answer_chain.invoke({"question": question})

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

        # Use hypothetical answer for retrieval instead of original question
        retriever = c.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(hypothetical_answer)

        # Now use original question with retrieved context for final answer
        rag_prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        final_chain = rag_prompt | self.llm.langchain_llm | StrOutputParser()

        formatted_context = format_docs(retrieved_docs)
        final_answer = final_chain.invoke(
            {"context": formatted_context, "question": question}
        )

        return {"answer": final_answer, "context": formatted_context}
