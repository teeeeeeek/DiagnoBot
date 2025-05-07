import os
import uuid
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from document_processor import split_text_into_chunks
from utils import check_api_key

def initialize_chroma_db(session_id):
    """
    Initialize a ChromaDB vector store.
    
    Args:
        session_id (str): Unique session identifier
        
    Returns:
        Chroma: Initialized vector store
    """
    # Check if OpenAI API key is available
    if not check_api_key():
        st.error("OpenAI API key is not set. Please set it to initialize the vector store.")
        return None
    
    try:
        # Create embeddings with OpenAI
        embeddings = OpenAIEmbeddings()
        
        # Create directory for vectorstore if it doesn't exist
        os.makedirs(f"chroma_db/{session_id}", exist_ok=True)
        
        # Initialize vector store
        vectorstore = Chroma(
            collection_name="medical_documents",
            embedding_function=embeddings,
            persist_directory=f"chroma_db/{session_id}"
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

def add_documents_to_vectorstore(vectorstore, texts, session_id):
    """
    Add documents to the vector store.
    
    Args:
        vectorstore (Chroma): Vector store instance
        texts (list): List of text documents
        session_id (str): Unique session identifier
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not vectorstore:
        return False
    
    try:
        # Process each text document
        for i, text in enumerate(texts):
            # Split text into chunks
            chunks = split_text_into_chunks(text)
            
            # Create Document objects
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"source": f"document_{i}", "chunk": j}
                )
                for j, chunk in enumerate(chunks)
            ]
            
            # Add documents to vector store
            vectorstore.add_documents(documents)
        
        # Persist the vector store
        vectorstore.persist()
        return True
    except Exception as e:
        st.error(f"Error adding documents to vector store: {str(e)}")
        return False

def load_existing_vectorstore(session_id):
    """
    Load an existing vector store from disk.
    
    Args:
        session_id (str): Unique session identifier
        
    Returns:
        Chroma: Loaded vector store or None if not found
    """
    # Check if vector store exists
    if not os.path.exists(f"chroma_db/{session_id}"):
        return None
    
    try:
        # Create embeddings with OpenAI
        embeddings = OpenAIEmbeddings()
        
        # Load vector store
        vectorstore = Chroma(
            collection_name="medical_documents",
            embedding_function=embeddings,
            persist_directory=f"chroma_db/{session_id}"
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None
