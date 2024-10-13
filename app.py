import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Functions to handle PDF text extraction and vector store creation
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss")

def ingest_data():
    # Check if the FAISS vector store already exists
    faiss_path = "Faiss"
    if os.path.exists(faiss_path):
        st.write("FAISS vector store already exists, skipping data ingestion.")
    else:
        st.write("Ingesting data for the first time, please wait...")
        pdf_files = [os.path.join("dataset", file) for file in os.listdir("dataset") if file.endswith(".pdf")]
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        create_vector_store(text_chunks)
        st.write("Data ingestion complete.")

def get_conversational_chain():
    prompt_template = """
    You are Lawy, a professional legal advisor specializing in Indian central laws. 
    Provide accurate, concise responses to user queries based on the Indian Legal Framework. 
    Always cite relevant legal sections and proceedings where applicable.
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0.3, 
        system_instruction="You are Lawy, a professional legal advisor specializing in Indian laws. Provide accurate advice and cite relevant legal sections.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("Faiss", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Redesigned hero section with more professional feel
def hero_section():
    st.markdown("""
    <style>
    .hero {
        padding: 2rem;
        background-color: #e8f0fe;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .hero h1 {
        color: #1b3a61;
        font-size: 3.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .hero p {
        color: #2f4f66;
        font-size: 1.25rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    .logo {
        height: 120px;
        margin-bottom: 1rem;
    }
    </style>
    
    <div class="hero">
        <img src="https://www.freeiconspng.com/uploads/legal-icon-png-27.png" class="logo">
        <h1>LawBot</h1>
        <p>Your Trusted AI Assistant for Indian Legal Laws.</p>
    </div>
    """, unsafe_allow_html=True)

# Main function with improved user interaction and design
def main():
    st.set_page_config("LawBot - Legal AI Assistant", page_icon=":scales:", layout="wide")
    
    # Call the updated hero section
    hero_section()

    if "data_ingested" not in st.session_state:
        st.session_state.data_ingested = False

    # Ingest data if not already done
    if not st.session_state.data_ingested:
        st.write("Checking for existing FAISS vector store...")
        ingest_data()
        st.session_state.data_ingested = True
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, I'm LawBot. How may I assist you with Indian Legal Acts today?"}]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['role'].title()}:** {message['content']}")

    # User input section
    prompt = st.chat_input("Type your legal question here...")
    
    # Handling user input and providing responses
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**User:** {prompt}")

        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal query..."):
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                response = user_input(prompt, chat_history)
                st.markdown(f"**LawBot:** {response}")


        # Append assistant's response to the session state
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

# Run the main function
if __name__ == "__main__":
    main()
