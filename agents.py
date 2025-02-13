from langchain.agents import create_react_agent, AgentExecutor
import tools
import model
import prompts


#### 1️⃣ BASIC GENERATOR ####
def generate_code_with_basic_agent(topic, language):
    """
    Generate code using a basic LangChain agent.

    Args:
        topic (str): Topic for the code snippet
        language (str): Programming language

    Returns:
        str: Generated code snippet
    """
    tool = tools.code_generator_tool()
    
    # Invoke the tool directly without an agent
    response = tool.func(topic, language)
    return response


#### 2️⃣ RAG ####
def retrieve_docs_with_rag(topic, vector):
    """
    Retrieve relevant documents using a RAG-based retriever.

    Args:
        topic (str): Topic to retrieve documents for
        vector (object): Vector store instance

    Returns:
        str: Retrieved documents
    """
    tool = tools.rag_retriever_tool(vector)
    
    # Invoke the tool directly without an agent
    response = tool.func(topic)
    return response


#### 3️⃣ AGENT ####
def generate_blog_with_agent(topic):
    """
    Generate a blog using a LangChain agent with Wikipedia as a knowledge source.

    Args:
        topic (str): Topic for the blog

    Returns:
        str: Generated blog content
    """
    tools_list = [tools.wikipedia_tool()]

    prompt_template = prompts.blog_generator_agent()
    llm = models.create_chat_groq_model()
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        handle_parsing_errors=True,
        verbose=True,
        stop_sequence=True,
        max_iterations=3
    )

    response = agent_executor.invoke({"input": topic})
    return response


#### 4️⃣ AGENT WITH RAG ####
def generate_blog_with_rag_agent(topic, vector):
    """
    Generate a blog using a LangChain agent with Retrieval-Augmented Generation (RAG).

    Args:
        topic (str): Topic for the blog
        vector (object): Instance of vector store

    Returns:
        str: Generated blog content
    """
    tools_list = [
        tools.blog_generator_tool(vector),
        tools.wikipedia_tool()
    ]

    prompt_template = prompts.blog_generator_agent_with_rag()
    llm = model.create_chat_groq_model()
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        handle_parsing_errors=True,
        verbose=True,
        stop_sequence=True,
        max_iterations=3
    )

    response = agent_executor.invoke({"input": topic})
    return response
