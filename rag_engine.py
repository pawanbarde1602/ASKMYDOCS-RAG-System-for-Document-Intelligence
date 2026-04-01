import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class AskMyDocsEngine:
    def __init__(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = "db"

    def process_pdf(self, pdf_path: str, chunk_strategy="fixed", chunk_size=1000, chunk_overlap=100):
        """
        Ingests a PDF and returns chunks based on the specified strategy.
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if chunk_strategy == "fixed":
            # Using RecursiveCharacterTextSplitter as it's more robust for general text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        elif chunk_strategy == "token":
            # Direct token count chunking as mentioned (500/1000/1500)
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
             # Default to recursive if unknown
             text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

        chunks = text_splitter.split_documents(documents)
        return chunks

    def create_vector_store(self, chunks):
        """
        Creates a Chroma vector store from document chunks.
        """
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def get_qa_chain(self, vectorstore, temperature=0.3):
        """
        Creates a QA chain with custom prompt engineering to reduce hallucinations.
        """
        # Optimized prompt for reducing hallucinations as requested
        template = """You are a professional AI assistant for private document Q&A.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and relevant to the provided context.

Context:
{context}

Question: {question}
Helpful Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        llm = ChatOpenAI(model_name="gpt-4", temperature=temperature)
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return qa_chain
