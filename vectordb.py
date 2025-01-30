import os
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ Ensure vectorstore is initialized properly
def initialize_chroma():
    """Initialize and return a new Chroma vector store instance."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="blog_chroma_db", embedding_function=embeddings)

# ✅ Store files in ChromaDB
def store_file_in_chroma(file, vectorstore):
    """Loads a file, splits its content, and stores it in ChromaDB."""
    
    # Save file temporarily
    temp_file_path = os.path.join("temp_storage", file.name)
    os.makedirs("temp_storage", exist_ok=True)  # Ensure directory exists

    with open(temp_file_path, "wb") as f:
        f.write(file.read())  # Save uploaded file to disk

    # Load document based on file type
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_file_path)
    elif file.name.endswith(".txt"):
        loader = TextLoader(temp_file_path)
    elif file.name.endswith(".csv"):
        loader = CSVLoader(temp_file_path)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_file_path)
    else:
        raise ValueError("Unsupported file format")

    docs = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Store in ChromaDB
    vectorstore.add_documents(split_docs)

    # Cleanup temporary file
    os.remove(temp_file_path)

# ✅ Retrieve documents from ChromaDB
def retrieve_from_chroma(query, vectorstore):
    """
    Retrieve documents from ChromaDB based on a query.

    Args:
        query (str): The query string.
        vectorstore (Chroma): The ChromaDB vector store instance.

    Returns:
        list: List of retrieved documents.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever.get_relevant_documents(query)
