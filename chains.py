from langchain_core.output_parsers import StrOutputParser
import models
import prompts
import vectordb

def generate_blog_chain(topic):
    """
    Generate a blog using a basic LLM prompt chain.

    Args:
        topic - Topic for the blog.

    Returns:
        str: Generated blog.
    """
    llm = models.create_chat_groq_model()
    prompt_template = prompts.blog_generator_prompt()
    chain = prompt_template | llm
    response = chain.invoke({"topic": topic})
    return response.content


def generate_blog_rag_chain(topic, vector):
    """
    Creates a RAG chain for retrieval and generation.

    Args:
        topic - Topic for retrieval.
        vectorstore - Instance of vector store.

    Returns:
        str: Generated blog.
    """
    prompt = prompts.blog_generator_rag_prompt()
    llm = models.create_chat_groq_model()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectordb.retrieve_from_chroma(topic, vectorstore=vector)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({
        "context": format_docs(retriever),
        "topic": topic
    })

    return response
