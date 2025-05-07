import os
import sys
import numpy as np
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from document_processor import extract_text_from_pdf, split_text_into_chunks
from utils import check_api_key, ensure_directories
from chat_handler import get_conversation_chain

# Ensure environment variables are set
os.environ["PYTHONIOENCODING"] = "utf-8"

def direct_load():
    # Set the session ID
    session_id = "fixed_session_001"
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if OpenAI API key is available
    if not check_api_key():
        print("Error: OpenAI API key is not available")
        return False
    
    # PDF file paths
    pdf_dir = "pdf_files"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return False
    
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    
    try:
        # Initialize embeddings
        print("Initializing embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Create directory for vectorstore
        print(f"Creating directory: chroma_db/{session_id}")
        os.makedirs(f"chroma_db/{session_id}", exist_ok=True)
        
        # Initialize vector store
        print("Initializing vector store...")
        vectorstore = Chroma(
            collection_name="medical_documents",
            embedding_function=embeddings,
            persist_directory=f"chroma_db/{session_id}"
        )
        
        # Process each PDF
        all_documents = []
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Extracting text from {pdf_file}...")
            text = extract_text_from_pdf(pdf_path)
            
            if not text:
                print(f"Failed to extract text from {pdf_file}")
                continue
                
            print(f"Successfully extracted {len(text)} characters from {pdf_file}")
            
            # Split text into chunks
            print(f"Splitting text into chunks...")
            chunks = split_text_into_chunks(text)
            print(f"Created {len(chunks)} chunks")
            
            # Create Document objects
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"source": pdf_file, "chunk": j}
                )
                for j, chunk in enumerate(chunks)
            ]
            
            all_documents.extend(documents)
        
        # Add all documents to vector store
        print(f"Adding {len(all_documents)} documents to vector store...")
        
        # We'll add documents in batches to prevent memory issues
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i+batch_size]
            print(f"Adding batch {i//batch_size + 1}/{len(all_documents)//batch_size + 1}...")
            vectorstore.add_documents(batch)
        
        # Persist the vector store
        print("Persisting vector store...")
        vectorstore.persist()
        
        # Save the session ID
        with open("session_id.txt", "w") as f:
            f.write(session_id)
        
        print("Successfully processed all documents!")
        return True
    
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return False

if __name__ == "__main__":
    success = direct_load()
    if success:
        print("\nYou can now use the Streamlit app to ask questions!")
    else:
        print("\nFailed to process documents. See errors above.")
