import os
import streamlit as st
from tempfile import NamedTemporaryFile
import time

from document_processor import extract_text_from_pdf
from embedding_manager import (
    initialize_chroma_db,
    add_documents_to_vectorstore,
    load_existing_vectorstore
)
from chat_handler import get_conversation_chain
from utils import check_api_key, get_session_id, ensure_directories

# Page configuration
st.set_page_config(
    page_title="MedStudy Assistant",
    page_icon="🩺",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_exists" not in st.session_state:
    st.session_state.vectorstore_exists = False
if "file_paths" not in st.session_state:
    st.session_state.file_paths = []
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

# Ensure necessary directories exist
ensure_directories()

# Get unique session ID
# First check if we have a fixed session ID from script processing
if os.path.exists("session_id.txt"):
    with open("session_id.txt", "r") as f:
        session_id = f.read().strip()
    st.session_state.using_fixed_session = True
else:
    session_id = get_session_id()
    st.session_state.using_fixed_session = False

st.session_state.vectorstore_exists = os.path.exists(f"chroma_db/{session_id}")

# Main page layout
st.title("MedStudy Assistant 🩺")

# Sidebar with file upload and processing
with st.sidebar:
    st.header("Document Management")
    
    # API Key input
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API key...",
        help="Get your API key from https://platform.openai.com/account/api-keys",
        value=os.environ.get("OPENAI_API_KEY", "")
    )
    
    # Set the API key in environment or session
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.divider()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your medical lecture notes or textbooks", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload PDF files containing your lecture notes or textbook chapters"
    )
    
    process_button = st.button("Process Documents")
    
    if process_button and uploaded_files:
        if not check_api_key():
            st.error("Please enter a valid OpenAI API key")
        else:
            with st.spinner("Processing your documents..."):
                # Save uploaded files to temp files
                temp_file_paths = []
                for uploaded_file in uploaded_files:
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_file_paths.append(tmp_file.name)
                
                # Extract text
                document_texts = []
                for file_path in temp_file_paths:
                    text = extract_text_from_pdf(file_path)
                    if text:
                        document_texts.append(text)
                    
                # Initialize vector store
                vectorstore = initialize_chroma_db(session_id)
                
                # Add documents to the vector store
                add_documents_to_vectorstore(vectorstore, document_texts, session_id)
                
                # Clean up temp files
                for file_path in temp_file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                st.session_state.files_processed = True
                st.session_state.vectorstore_exists = True
                
                # Initialize conversation chain
                st.session_state.conversation = get_conversation_chain(session_id)
                
                st.success("Documents processed successfully!")
    
    # Show status
    if st.session_state.files_processed:
        st.success("✅ Documents loaded and processed")
    elif st.session_state.vectorstore_exists:
        st.info("📚 Using previously processed documents")
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(session_id)

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Reset everything button
    if st.button("Reset Everything"):
        # Clear session state
        st.session_state.chat_history = []
        st.session_state.conversation = None
        st.session_state.files_processed = False
        
        # Remove vectorstore directory
        if os.path.exists(f"chroma_db/{session_id}"):
            import shutil
            shutil.rmtree(f"chroma_db/{session_id}")
        
        st.session_state.vectorstore_exists = False
        st.rerun()

    st.divider()
    st.caption("MedStudy Assistant helps you learn medical concepts more efficiently by answering questions based on your own materials.")

# Main chat interface
if not check_api_key():
    st.warning("Please enter your OpenAI API key in the sidebar to get started.")
elif not st.session_state.vectorstore_exists:
    st.info("👈 Please upload and process your medical documents to get started.")
    
    # Debug information
    with st.expander("Debug Information"):
        st.write(f"Session ID: {session_id}")
        st.write(f"Vector store exists: {st.session_state.vectorstore_exists}")
        st.write(f"Vector store path: chroma_db/{session_id}")
        st.write(f"Files processed: {st.session_state.files_processed}")
        st.write(f"API key present: {bool(os.environ.get('OPENAI_API_KEY'))}")
        
        # Try to verify the API key
        api_verified = False
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            models = client.models.list()
            api_verified = True
        except Exception as e:
            st.error(f"API verification error: {str(e)}")
            
        st.write(f"API key verified: {api_verified}")
    
    # Show examples
    st.header("Example Questions You Can Ask:")
    example_cols = st.columns(2)
    with example_cols[0]:
        st.markdown("- What are the key symptoms of Crohn's disease?")
        st.markdown("- Explain the cardiac conduction system step by step.")
    with example_cols[1]:
        st.markdown("- Compare and contrast Type 1 and Type 2 diabetes.")
        st.markdown("- Summarize the mechanism of action for ACE inhibitors.")
else:
    # Initialize conversation if needed
    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(session_id)
        
    # Debug information
    with st.expander("Debug Information"):
        st.write(f"Session ID: {session_id}")
        st.write(f"Vector store exists: {st.session_state.vectorstore_exists}")
        st.write(f"Vector store path: chroma_db/{session_id}")
        st.write(f"Files processed: {st.session_state.files_processed}")
        st.write(f"Conversation initialized: {st.session_state.conversation is not None}")
        st.write(f"Chat history items: {len(st.session_state.chat_history)}")
        
        if os.path.exists(f"chroma_db/{session_id}"):
            try:
                import glob
                files = glob.glob(f"chroma_db/{session_id}/*")
                st.write(f"Files in vectorstore: {files}")
            except Exception as e:
                st.error(f"Error listing vectorstore files: {str(e)}")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🩺"):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your medical notes..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="🩺"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get response from the chain
            try:
                if st.session_state.conversation is None:
                    # Try to recreate the conversation chain
                    st.session_state.conversation = get_conversation_chain(session_id)
                    
                    if st.session_state.conversation is None:
                        error_msg = "Cannot create conversation chain. Please make sure you've processed documents first."
                        message_placeholder.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    else:
                        with st.spinner("Thinking..."):
                            response = st.session_state.conversation({"question": prompt})
                            full_response = response["answer"]
                        
                        message_placeholder.write(full_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                else:
                    with st.spinner("Thinking..."):
                        response = st.session_state.conversation({"question": prompt})
                        full_response = response["answer"]
                    
                    message_placeholder.write(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    # Show features coming soon
    with st.expander("Features Coming Soon"):
        st.markdown("""
        - **Quiz Generation**: Create custom quizzes based on your materials
        - **Flashcards**: Generate flashcards for spaced repetition learning
        - **Mnemonics**: Get memory aids for complex medical concepts
        - **Mind Maps**: Visualize connections between medical concepts
        - **Summary Generator**: Create concise summaries of lengthy texts
        """)
