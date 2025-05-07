import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from embedding_manager import load_existing_vectorstore

def get_llm():
    """
    Get the language model for chat completions.
    
    Returns:
        ChatOpenAI: Configured language model
    """
    # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    # Do not change this unless explicitly requested by the user
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        verbose=True
    )

def get_conversation_chain(session_id):
    """
    Create a conversational chain with retrieval capabilities.
    
    Args:
        session_id (str): Unique session identifier
        
    Returns:
        ConversationalRetrievalChain: Configured conversation chain
    """
    try:
        # Load vector store
        vectorstore = load_existing_vectorstore(session_id)
        
        if not vectorstore:
            st.error("No vector store found. Please process documents first.")
            return None
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Get language model
        llm = get_llm()
        
        # Create retriever with search params
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5},  # Retrieve top 5 most relevant chunks
            search_type="similarity"
        )
        
        # Create conversation chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            return_source_documents=False
        )
        
        # Add medical context system prompt
        chain.combine_docs_chain.llm_chain.prompt.messages[0].content = """
        You are MedStudy Assistant, a helpful AI tutor specialized in medical education. 
        You're here to help medical students understand complex concepts, explain medical 
        terminology clearly, and provide accurate information based on their uploaded materials.
        
        When answering questions:
        1. Respond in a friendly, supportive tone like a knowledgeable peer or tutor
        2. Use clear, concise explanations with medical accuracy
        3. Include relevant anatomical or physiological details when appropriate
        4. Structure complex answers with bullet points or numbered lists when helpful
        5. Only provide information from the retrieved context - don't make up information
        6. If you don't know the answer or can't find it in the context, be honest and suggest alternatives
        
        The retrieved context below contains information from the student's own lecture notes and textbooks.
        Use this information to provide personalized, accurate responses.
        """
        
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None
