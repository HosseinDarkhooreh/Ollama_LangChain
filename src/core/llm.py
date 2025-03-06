import logging
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma


logger = logging.getLogger(__name__)

class LLMManager:
        
    def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
            """
            Process a user question using the vector database and selected language model.
            """
            logger.info(f"Processing question: {question} using model: {selected_model}")
            
            # Initialize LLM
            llm = ChatOllama(model=selected_model)
            
            # Query prompt template
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI assistant. Please generate 3
                different variation of the given user question in order to retrieve more relevant documents from
                a vector database. By generating multiple perspectives on the given question, your
                goal is to help the user overcome some of the shortcoming of the distance-based
                similarity search. Provide these different questions separated by newlines.
                Original question: {question}""",
            )

            # Set up retriever
            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(),
                llm,
                prompt=QUERY_PROMPT
            )

            # RAG prompt template
            template = """Answer the question based ONLY on the following context and STRICTLY in German:
            {context}
            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)

            # Create chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            response = chain.invoke(question)
            logger.info("Question processed and response generated")
            return response