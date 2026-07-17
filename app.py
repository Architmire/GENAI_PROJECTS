from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# creating my prompt 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant please respond to questions asked"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title("My GPT")
input_text = st.text_input("Enter your question here:")

# Ollama framework along with gamma2:latest model 
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# GPT output
if input_text:
    st.write(chain.invoke({"question": input_text}))
