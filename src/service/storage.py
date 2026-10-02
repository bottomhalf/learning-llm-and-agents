from typing import Optional

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


class Storage:
    def __init__(self):
        self.name = 'Storage'
        self.embedding_model_name = "BAAI/bge-base-en-v1.5"
        self.embedding: HuggingFaceEmbeddings = None
        self.store: Optional[Chroma] = None
        self.retrieval: Optional[VectorStoreRetriever] = None

    def start_setup(self):
        # setup embedding function
        self.embedding = self.init_huggingface()

        # setup vector store - chroma db
        self.store = self.init_chroma()

        # setup retrieval
        self.retrieval = self.get_retrieval()

    def find_related_chunks(self, user_prompt):
        docs = self.retrieval.invoke(user_prompt)
        return docs

    def init_huggingface(self):
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

    def embedd_text(self, text):
        return self.embedding.embed_documents(text)

    def add_document_in_store(self, docs):
        self.store.add_documents(docs)

    def init_chroma(self):
        return Chroma(
            collection_name="dream",
            persist_directory="./src/doc/store",
            embedding_function=self.embedding
        )

    def get_retrieval(self):
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            }
        )
