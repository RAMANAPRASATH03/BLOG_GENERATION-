import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # Hugging Face for embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
import os

# Initialize ChromaDB with Hugging Face Embeddings for Blog Content
def initialize_chroma():
    """
    Initializes ChromaDB instance with Hugging Face embeddings.

    Returns:
        Chroma: Vector database instance for blog content storage.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="blog_chroma_db", embedding_function=embeddings)

def retrieve_blog_content(topic, vectorstore):
    """
    Retrieves relevant blog content from ChromaDB.

    Args:
        topic (str): The blog topic for retrieval.
        vectorstore (Chroma): The vector database instance.

    Returns:
        List[Document]: Retrieved documents relevant to the blog topic.
    """
    retriever = vectorstore.as_retriever()
    return retriever.get_relevant_documents(topic)

def store_blog_content_in_chroma(file, vectorstore):
    """
    Ingests a document (PDF, TXT, CSV, DOCX) into ChromaDB for blog content generation.

    Args:
        file: Uploaded file object containing blog-related content.
        vectorstore (Chroma): The vector database instance.
    """
    file_path = f"blog_temp_files/{file.name}"
    os.makedirs("blog_temp_files", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file.read())

    # Load document based on file type
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.name.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file.name.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    docs = loader.load()

    # Split documents into manageable chunks for vector storage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    vectorstore.add_documents(split_docs)
