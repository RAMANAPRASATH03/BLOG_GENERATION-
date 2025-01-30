import streamlit as st
import chains
import vectordb
import agents

def blog_generator_app():
    """
    Generates Blog Outline Generator App with Streamlit, providing user input and displaying output.
    Includes a sidebar with two sections: Blog Generator and File Ingestion for RAG.
    """

    # Custom CSS for vibrant UI
    st.markdown(
        """
        <style>
            .main {background-color: #f0f8ff;}
            .stButton button {background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px 20px;}
            .stTextInput {background-color: #e6f7ff;}
            .stSidebar {background-color: #1e3a8a; color: white;}
            .stSidebar input {background-color: #f8fafc; color: #1e3a8a;}
            .stSidebar h2 {color: #fbbf24;}
            .st-info-box {background-color: #f0fdfa; border-radius: 15px; padding: 20px; color: #064e3b;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    st.sidebar.title("Menu")
    section = st.sidebar.radio(
        "Choose a section:",
        ("Blog Generator RAG", "RAG File Ingestion")
    )

    # db initialization
    vectordatabase = vectordb.initialize_chroma()

    # Condition for blog generation page
    if section == "Blog Generator RAG":
        st.title("Generate a Blog Outline! ✍️")

        with st.form("blog_generator"):
            topic = st.text_input(
                "Enter a topic for the blog:",
                placeholder="e.g., The Future of AI"
            )
            submitted = st.form_submit_button("✨ Generate Outline ✨")

            is_rag_enabled = st.checkbox("Enable RAG")
            is_agent_enabled = st.checkbox("Enable Agent")

            if submitted:
                with st.spinner('Generating your blog outline...'):
                    if is_rag_enabled and is_agent_enabled:
                        response = agents.generate_blog_with_rag_agent(topic, vectordatabase)
                    elif is_agent_enabled:
                        response = agents.generate_blog_with_agent(topic)
                    elif is_rag_enabled:
                        response = chains.generate_blog_rag_chain(topic, vectordatabase)
                    else:
                        response = chains.generate_blog_chain(topic)

                    st.markdown(
                        f"""
                        <div class="st-info-box">
                            <h3>📝 Blog Outline</h3>
                            <p>{response}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Condition for RAG File Ingestion
    elif section == "RAG File Ingestion":
        st.title("RAG File Ingestion 📂")

        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv", "docx", "pdf"])

        if uploaded_file is not None:
            vectordb.store_file_in_chroma(uploaded_file, vectordatabase)
            st.success(f"File '{uploaded_file.name}' uploaded and embeddings stored in vectordb successfully!")

blog_generator_app()
