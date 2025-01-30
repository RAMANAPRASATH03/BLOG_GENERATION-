# utils.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def process_pdf_for_rag(uploaded_file):
    temp_file_path = f"temp_{uploaded_file.name}"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        return splits
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
