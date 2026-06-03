
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

st.title("Financial AI Research Assistant")

pdf = st.file_uploader("Upload financial report", type=["pdf"])

if pdf:
    with open("temp.pdf","wb") as f:
        f.write(pdf.read())

    docs = PyPDFLoader("temp.pdf").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=qa_pipe)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    query = st.text_input("Ask a question about the report")
    if query:
        st.write(qa.run(query))
