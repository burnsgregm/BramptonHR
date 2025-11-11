
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import time

# --- 1. Global Configurations (Mirrored from Colab) ---

# We get the API key from Streamlit's secrets
# You must set this in your app's settings on share.streamlit.io
GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GEMINI_API_KEY secret not found. Please set it in your Streamlit app settings.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Use the exact model names that worked in Colab
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_LLM_MODEL = "models/gemini-2.5-pro" # Using the model you confirmed
APP_TITLE = "Brampton L&D Co-Pilot (PoC)"

# Paths for documents and vector store
DOCS_DIR = "docs"
VECTORSTORE_PATH = "vectorstore" # This app will load the pre-built store

RAG_PROMPT_TEMPLATE = """
SYSTEM INSTRUCTION:
You are the "AI Mentor," a helpful and safe AI assistant for City of Brampton employees.
Your job is to answer questions *only* based on the official, approved policy documents provided as context.
DO NOT, under any circumstances, use any outside knowledge or make up information.

Your answer MUST follow these rules:
1.  **Source Restriction:** Base your answer *only* on the text provided in the "<context>" section.
2.  **No Outside Knowledge:** If the answer is not in the context, you MUST state: "I'm sorry, I do not have access to that specific information in the approved policy documents. Please contact your HR Advisor."
3.  **Proactive "Golden Thread":** If (and *only* if) the user's question is about "hiring," "recruiting," or "job posting," you MUST first answer their question, and *then* you MUST add the following proactive nudge:
    "Additionally, please be aware that Policy HRM-160 requires all hiring committee members to complete the 'Mandatory Recruitment and Diversity Learning Series.' You can find this module on the L&D portal."
4.  **Tone:** Be professional, helpful, and concise.

<context>
{context}
</context>

Question: {input}

HELPFUL ANSWER (Based *only* on the context):
"""

# --- 2. Caching & RAG Chain Functions ---

# Use Streamlit's cache to load models and data only once
@st.cache_resource
def load_models():
    """Loads the embedding model and LLM."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
        llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL)
        return embeddings, llm
    except Exception as e:
        st.error(f"Error loading AI models: {e}")
        st.stop()

@st.cache_resource
def load_vector_store(_embeddings):
    """
    Loads the pre-built FAISS vector store from the repo.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        st.error(f"Vector store not found at {VECTORSTORE_PATH}. Please ensure the 'vectorstore' directory is in the GitHub repo.")
        st.stop()
        
    try:
        # Load the pre-built index from local disk
        db = FAISS.load_local(VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.stop()


@st.cache_resource
def create_rag_chain(_llm, _retriever):
    """Creates the final RAG chain."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    document_chain = create_stuff_documents_chain(_llm, prompt)
    rag_chain = create_retrieval_chain(_retriever, document_chain)
    return rag_chain

# --- 3. Streamlit App UI ---

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🤖 {APP_TITLE}")
st.markdown("This prototype is a demonstration for the **City of Brampton AI PoC Program**. It is an 'AI Mentor' that answers questions based *only* on the 6 official, public-facing HR policy documents.")

# Load models and data
embeddings, llm = load_models()
db = load_vector_store(embeddings)
retriever = db.as_retriever()
rag_chain = create_rag_chain(llm, retriever)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I am the AI Mentor. How can I help you with Brampton's HR policies today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question (e.g., 'What is the policy on harassment?')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display AI response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            full_response = response['answer']
            st.markdown(full_response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- This print statement is AFTER the magic command ---
print(f"Wrote app.py to {REPO_PATH}/app.py")
print("\nSection 3.0 complete.")
