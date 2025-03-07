
# Local RAG with  Ollama, LangChain, Chromo DB


## Prerequisites

1. **Install Ollama**
   - Visit [Ollama's website](https://ollama.com) to download and install
   - Pull required models (LLM and Embedding):
     - e.g. ollama pull llama3.2      # for LLM Model
     - ollama pull nomic-embed-text   # for Embedding Model
     ```

2. **Set Up Virtual Environment**
   create a virtual environment using tools such as venv, pipenv, etc..
   Install required packages inside the virtual environment with pip install -r requirements.txt
  
## Running the Application

python run.py

Then open your browser with url `http://localhost:8501` if it did not open automatically

## Notes
- Ollama should automatically detect and use the GPU to run models.
- This quick and simple solution to demonstrate capability of Ollama and LangChain. Later maybe can be tested with other databases such as Qdrant, Weaviate, etc..
