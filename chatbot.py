import os
import streamlit as st
from dotenv import load_dotenv


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# -------------------------
# Load API Key
# -------------------------
load_dotenv()  # this will load variables from a .env file into environment


groq_api_key = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="C++ Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 C++ Chatbot")
st.write("Ask anything about C++")

# -------------------------
# Load Documents
# -------------------------
@st.cache_resource
def load_vector_db():

    loader = TextLoader(
        "cppTextData.txt",
        encoding="utf-8"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        docs,
        embeddings
    )

    return db


db = load_vector_db()

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------
# Retrieval Chain
# -------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True
)

# -------------------------
# User Input
# -------------------------
question = st.text_input("Enter your question")

if st.button("Ask"):

    if question:

        with st.spinner("Thinking..."):

            result = qa.invoke({"query": question})

            st.subheader("Answer")

            st.write(result["result"])

            with st.expander("Relevant Context"):

                for i, doc in enumerate(result["source_documents"]):

                    st.write(f"### Chunk {i+1}")

                    st.write(doc.page_content)

    else:

        st.warning("Please enter a question.")