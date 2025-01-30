from langchain.agents import Tool
import chain
from vectordb import retrieve_from_chroma
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


def code_generator_tool():
    """
    Generate a tool that can create code snippets.

    Returns:
        Tool: A LangChain Tool object for code generation.
    """
    return Tool(
        name="Code Generator",
        func=lambda topic, language: chain.generate_code_chain(topic, language),
        description="Generates a code snippet based on a given topic and programming language."
    )


def rag_retriever_tool(vector):
    """
    Create a Tool for retrieving relevant documents using RAG.

    Args:
        vector (object): The vector store instance.

    Returns:
        Tool: A LangChain Tool object for RAG retrieval.
    """
    return Tool(
        name="RAG Retriever",
        func=lambda topic: "\n\n".join(
            doc.page_content for doc in retrieve_from_chroma(topic, vectorstore=vector)
        ),
        description="Retrieves relevant documents for a given topic using a vector store."
    )


def wikipedia_tool():
    """
    Create a Tool for retrieving Wikipedia content.

    Returns:
        Tool: A LangChain Tool object for Wikipedia search.
    """
    return Tool(
        name="Wikipedia Search",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Fetches relevant information from Wikipedia based on a given topic."
    )


def blog_generator_tool(vector):
    """
    Create a Tool for generating blog content.

    Args:
        vector (object): The vector store instance for RAG retrieval.

    Returns:
        Tool: A LangChain Tool object for blog generation.
    """
    def generate_blog(topic):
        # Retrieve Wikipedia information
        wiki_summary = wikipedia_tool().func(topic)

        # Retrieve RAG documents
        rag_documents = rag_retriever_tool(vector).func(topic)

        # Combine content for blog generation
        blog_content = f"### {topic}\n\n"
        blog_content += f"#### Wikipedia Summary:\n{wiki_summary}\n\n"
        blog_content += f"#### Additional Insights from RAG:\n{rag_documents}\n\n"
        blog_content += "This blog combines structured knowledge from Wikipedia and relevant documents retrieved using RAG."

        return blog_content

    return Tool(
        name="Blog Generator",
        func=generate_blog,
        description="Generates a blog post using Wikipedia summaries and RAG-retrieved documents."
    )
