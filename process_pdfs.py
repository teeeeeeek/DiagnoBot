import os
import sys
import uuid
from document_processor import extract_text_from_pdf
from embedding_manager import initialize_chroma_db, add_documents_to_vectorstore
from utils_script import ensure_directories, check_api_key

def process_pdfs(pdf_dir):
    # Ensure directories exist
    ensure_directories()
    
    # Create a fixed session ID for consistency
    session_id = "medstudyasst_001"
    
    # Initialize vectorstore
    print("Initializing vector store...")
    vectorstore = initialize_chroma_db(session_id)
    
    if not vectorstore:
        print("Failed to initialize vector store. Check OpenAI API key.")
        return False
    
    # Get all PDFs in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return False
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    all_texts = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Extracting text from {pdf_file}...")
        text = extract_text_from_pdf(pdf_path)
        if text:
            all_texts.append(text)
            print(f"Successfully extracted text from {pdf_file} ({len(text)} characters)")
        else:
            print(f"Failed to extract text from {pdf_file}")
    
    if not all_texts:
        print("No text was extracted from any PDF files")
        return False
    
    # Add documents to vector store
    print("Adding documents to vector store...")
    result = add_documents_to_vectorstore(vectorstore, all_texts, session_id)
    
    if result:
        print("Successfully processed all documents")
        # Write the session_id to a file so the app can use it
        with open("session_id.txt", "w") as f:
            f.write(session_id)
        return True
    else:
        print("Failed to add documents to vector store")
        return False

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable must be set")
        sys.exit(1)
        
    pdf_dir = "pdf_files"
    success = process_pdfs(pdf_dir)
    
    if success:
        print("PDFs processed successfully!")
        print("You can now run the Streamlit app and ask questions about your documents.")
    else:
        print("Failed to process PDFs. Check the errors above.")
