import os
import uuid
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
    except Exception as e:
        print(f"API key validation error: {e}")
        return False

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
