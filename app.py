# Importing necessary modules and classes
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import QdrantClient
import os

os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
#os.environ['GOOGLE_API_KEY'] = "AIzaSyDrCgz1QdzYdm5TNbpBXjTf9Y7Uy0BybfI"

# Initialize LLM and embedding model
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Load and process PDF document
file_path = "data/grow_rich.pdf"
loader = PyPDFLoader(file_path)
doc = loader.load()
document = doc[18:]  # Skipping some front pages
# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(document)
# Load Qdrant credentials
url = st.secrets["qdrant"]["url"]
api_key = st.secrets["qdrant"]["api_key"]
#url="https://647615bf-9468-4f8e-9c22-66f7929e14f3.europe-west3-0.gcp.cloud.qdrant.io:6333"
#api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.d34m2ZcvzmfKPl3H2z9ge_Oy4Yc3v4PuhLbUeJ_XHgk"

# ⚠️ Deleting existing collection (use only when needed)
client = QdrantClient(url=url, api_key=api_key)
client.delete_collection(collection_name="first_document")

# Adding new collection
qdrant = QdrantVectorStore.from_documents(
    documents=doc,
    embedding=embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="first_document"
)

# Set up retriever
retriever = qdrant.as_retriever(search_kwargs={"k": 50})
# qa_chain
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
  # Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.set_page_config(page_title='RAG Chatbot', page_icon="🤖")
st.title("Ask *Think and Grow Rich*")
st.markdown("This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions from the book *Think and Grow Rich*.")

# Input from user
query = st.chat_input("📩 Ask a question:")
if query:
    with st.spinner("Searching for the best answer..."):
        result = qa_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((query, result["answer"]))
        st.success(result["answer"])

# Show chat history
if st.session_state.chat_history:
    st.write("### 💬 Conversation History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")