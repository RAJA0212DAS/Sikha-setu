from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(path):
    docs = []

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())

    return docs