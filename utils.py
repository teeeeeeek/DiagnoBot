import os
import uuid
import streamlit as st
from openai import OpenAI

def check_api_key():
    """
    Check if a valid OpenAI API key is available.
    
    Returns:
        bool: True if valid API key is found, False otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        # Test API key with a simple request
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def get_session_id():
    """
    Get a unique session ID for the current user session.
    
    Returns:
        str: Unique session ID
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    return st.session_state.session_id

def ensure_directories():
    """
    Ensure necessary directories exist.
    """
    os.makedirs("chroma_db", exist_ok=True)

def sanitize_filename(filename):
    """
    Sanitize a filename to be safe for file systems.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace potentially unsafe characters
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        filename = filename.replace(char, '_')
    
    return filename

def format_chat_message(message, is_user=False):
    """
    Format a chat message for display.
    
    Args:
        message (str): Message content
        is_user (bool): Whether the message is from the user
        
    Returns:
        str: Formatted message HTML
    """
    if is_user:
        return f"""
        <div class="chat-message user-message">
            <div class="avatar">👤</div>
            <div class="message">{message}</div>
        </div>
        """
    else:
        return f"""
        <div class="chat-message bot-message">
            <div class="avatar">🩺</div>
            <div class="message">{message}</div>
        </div>
        """
