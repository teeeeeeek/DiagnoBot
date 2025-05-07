import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """
    Split text into chunks for processing.
    
    Args:
        text (str): Text to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

def process_document(pdf_path):
    """
    Process a PDF document by extracting text and splitting into chunks.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of text chunks
    """
    text = extract_text_from_pdf(pdf_path)
    if text:
        chunks = split_text_into_chunks(text)
        return chunks
    return []
