import datetime
import os
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from src.service.storage import Storage
from dotenv import load_dotenv
load_dotenv()


def clean_text(text):
    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z\s]", "", text).strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


class SimpleLangchain:
    def __init__(self):
        self.fileName = 'SimpleLangchain'
        self.filePath = './src/doc/dream.txt'
        self.documents: list[Document] = []
        self.storage = Storage()
        self.storage.start_setup()
        self.llm_model_name = "openai/gpt-oss-120b"

        self.llm = self.init_groq_llm()

    def setup_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """
            Please provide a polite and help response to the following question using provided context.
            Ensure that the tone remains professional, courteous, and empathetic and tailor the response.

            ### Context:
            {context}

            ### Question:
            {question}

            ### Polite Response:
            In you response, consider including:
            - Acknowledge the use's query and express gratitude for the opportunity
            - Provide a clear and concise answer tht directly address the question
            - Use positive language and maintain a supportive tone throughout.
            - If applicable, include relevant information or resources that could support the response
            - Conclude by inviting any follow-up questions or providing encourage to ask next question 
            """
        )
        chain = (
            {
                "context": self.storage.retrieval | RunnableLambda(
                    lambda docs: "\n\n".join(d.page_content for d in docs)
                ),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def init_groq_llm(self):
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=self.llm_model_name,
            temperature=0.8
        )

    def get_answer(self, query):
        chain = self.setup_chain()
        result = chain.invoke(query)
        print(f"Response: {result}")

    def document_loader(self):
        print(f"Hi, this is Simple {self.fileName} class")
        loader = TextLoader(self.filePath)
        self.documents = loader.load()

    def split_document(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )

        # splitter.split_documents() returns a list[Document]
        documents: list[Document] = splitter.split_documents(self.documents)

        # Retain Document objects with cleaned text and original metadata
        cleaned_documents: list[Document] = [
            Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
            for doc in documents
        ]

        document_id = str(uuid.uuid4())
        for index, doc in enumerate(cleaned_documents):
            # embedded_text = self.storage.embedd_text(doc.page_content)
            print(f"Page index[{index}], Content: {doc.page_content}")
            doc.metadata = {
                "document_id": document_id,
                "chunk_id": index,
                "source_file": "dream.txt",
                "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

        self.storage.add_document_in_store(cleaned_documents)
        return cleaned_documents
