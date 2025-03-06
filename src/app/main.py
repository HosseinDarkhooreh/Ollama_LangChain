
"""
Streamlit application for PDF-based RAG) using Ollama + LangChain + Streamlit

Running the Streamlit application
"""
import os
import streamlit as st
import shutil
import pdfplumber
import logging
import logging.config
import yaml


from pathlib import Path
from src.core.loader import FileLoader
from src.core.embedder import VectorStore
from src.core.utils import Utils
from src.core.llm import LLMManager
from src.core.constants import DirectoryPath



# Streamlit page configuration
st.set_page_config(
    page_title="Ollama RAG With LangChain and Streamlit UI",
    page_icon="☀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main() -> None:
    """
    Main function to run the Streamlit application.
    """

# Parsing YML file
    
    CONFIG_FILE = DirectoryPath.CONFIG_FILE.value
    PERSIST_DIRECTORY = DirectoryPath.CHROMA_PERSIST_DIRECTORY.value
    
    
    config = CONFIG_FILE
    config = yaml.safe_load(open(config))

    # configure logging
    log_config = config['logging']
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__name__)
    
    # reading chunk configuration
    chunk_config = config['chunk']
    
    # reading AI model configuration
    model_config = config['model']
    
    # reading vector dbs configuration
    vector_db_config = config['vector_db']

  
    st.subheader("Ollama LangChain RAG", divider="gray", anchor=False)

    
    # Create layout
    col_left, col_right = st.columns([1.5, 2])
    col_embedding_model,col_vector_db = col_left.container().columns(2)
    col_file_uploader = col_left.container()
    col_vectrorizer,col_delete_collection = col_left.container().columns(2)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "file_upload" not in st.session_state:
        st.session_state["file_upload"] = None
    if "temp_dir" not in st.session_state:
        st.session_state["temp_dir"] = None
    if "chunked_text" not in st.session_state:
        st.session_state["chunked_text"] = None
   

    # Get llm models
    available_llm_models = model_config['llm_model']
    # Get embedding models
    available_embedding_models = model_config['embedding_model']
    # Get vector dbs
    available_vector_dbs = vector_db_config['db_name']
    
      
      # Model selection
    if available_llm_models:
        selected_llm_model = col_right.selectbox(
            "Pick a model available locally on your system", 
            available_llm_models,
            key="llm_model_select"
        )


    # Embedding model selection
    if available_embedding_models:
        with col_embedding_model:
            selected_embedding_model = col_embedding_model.selectbox(
            "Pick an embedding model", 
            available_embedding_models,
            key="embedding_model_select"
            )
    

    # Vector DB selection
    if available_vector_dbs:
        with col_vector_db:
            selected_vector_db = col_vector_db.selectbox(
            "Pick a vector database", 
            available_vector_dbs,
            key="vector_db_select"
            )
    
    # Regular file upload with unique key
    with col_file_uploader:
        file_upload = col_file_uploader.file_uploader(
        "Upload a PDF file", 
        type="pdf", 
        accept_multiple_files=False,
        key="pdf_uploader"
        )


    if file_upload:
        st.session_state["file_upload"] = None
        st.session_state["chunked_text"] = None
        if st.session_state["temp_dir"] and Path(st.session_state["temp_dir"]).is_dir():
            shutil.rmtree(st.session_state["temp_dir"])
            st.session_state["temp_dir"] = None
        with st.spinner("Chunking uploaded PDF..."):
            file_loader= FileLoader(chunk_size=chunk_config['chunk_size'],chunk_overlap=chunk_config['chunk_overlap'])
            st.session_state["chunked_text"], st.session_state["temp_dir"] = file_loader.load_split(file_upload)
            
            # Store the uploaded file in session state
            st.session_state["file_upload"] = file_upload
            # Extract and store PDF pages
            with pdfplumber.open(file_upload) as pdf:
                st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col_file_uploader.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=600, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col_file_uploader:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)


    # Vectorize PDF button
    with col_vectrorizer:
        vectorize_pdf = col_vectrorizer.button(
        "🎲 Vectorize PDF", 
        type="secondary",
        key="vectorize_button"
        )

    if vectorize_pdf:
        if "file_upload" in st.session_state and st.session_state["file_upload"]:
            st.session_state.pop("vector_db", None)
            embedding_model= VectorStore(embedding_model=selected_embedding_model)
            st.session_state["vector_db"]= embedding_model.create_vector_db(st.session_state["file_upload"],st.session_state["chunked_text"],PERSIST_DIRECTORY)
            st.session_state["file_upload"] = None
            if st.session_state["temp_dir"]:
                shutil.rmtree(st.session_state["temp_dir"])
            
    # Delete collection button
    with col_delete_collection:
        delete_collection = col_delete_collection.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
        )

  
    if delete_collection:
        if st.session_state["temp_dir"] and Path(st.session_state["temp_dir"]).is_dir():
            shutil.rmtree(st.session_state["temp_dir"])
        VectorStore.delete_vector_db(st.session_state["vector_db"])
        # os.rename(PERSIST_DIRECTORY, PERSIST_DIRECTORY)
        shutil.rmtree(PERSIST_DIRECTORY)
            

    # Chat interface
    with col_right:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "👽" if message["role"] == "assistant" else "🙂"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter your question here...", key="chat_input"):
            try:
                if st.session_state["temp_dir"] and Path(st.session_state["temp_dir"]).is_dir():
                    shutil.rmtree(st.session_state["temp_dir"])
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="🙂"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="👽"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = LLMManager.process_question(
                                prompt, st.session_state["vector_db"], selected_llm_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")


if __name__ == "__main__":
    main()
