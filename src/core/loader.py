"""Loading PDF and Splitting."""
import logging
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class FileLoader:
   
    
    def __init__(self,chunk_size: int=7500, chunk_overlap: int=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temp_dir=None
            
    def load_split(self, file_upload: str):
        """
        Load PDF file and split into chunks.

        """
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        
        logger.info(f"Creating split from file upload: {file_upload.name}")
       
        self.temp_dir = tempfile.mkdtemp()

        temp_path = os.path.join(self.temp_dir, file_upload.name)
        with open(temp_path, "wb") as f:
            f.write(file_upload.getvalue())
            logger.info(f"File saved to temporary path: {temp_path}")
            loader = PyMuPDFLoader(temp_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(data)
            logger.info("Document split into chunks")
            f.close()
            return chunks, self.temp_dir