"""Vector embeddings and database functionality."""
import os
import logging
import streamlit as st
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector embeddings and database operations."""
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_db = None
    
    def create_vector_db(self, file_name, chunks: List, persist_directory: str) -> Chroma:
        """Create vector database from chunked text.
        """
       
        try:
            logger.info("Creating vector database")
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=os.path.join(persist_directory),
                collection_name=f"pdf_{hash(file_name.name)}"  # Unique collection name per file
            )
            logger.info("Vector DB created with persistent storage")
            return self.vector_db
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise

  


    def delete_vector_db(vector_db: Optional[Chroma]) -> None:
        """
        Delete the vector database and clear related session state.

        """
        logger.info("Deleting vector DB")
        if vector_db is not None:
            try:
                                
                # Clear session state
                st.session_state.pop("pdf_pages", None)
                st.session_state.pop("file_upload", None)
                st.session_state.pop("vector_db", None)
                st.session_state.pop("temp_dir", None)
                st.session_state.pop("chunked_text", None)

                # Delete the collection
                
                vector_db.delete_collection()
                
                st.success("Collection and temporary files deleted successfully.")
                logger.info("Vector DB and related session state cleared")
                # st.rerun()
            except Exception as e:
                st.error(f"Error deleting collection: {str(e)}")
                logger.error(f"Error deleting collection: {e}")
        else:
            st.error("No vector database found to delete.")
            logger.warning("Attempted to delete vector DB, but none was found")
