from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#creating my prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpfull assistant. please respond to the question asked"),
        ("user","Question:{questions}")
    ]
)

#Streamlit framework
st.title("My GPT")
input_text = st.text_input("What question do you have in the mind :")

#Ollama framework along with gemma2:2b
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

#GPT utput
if input_text:
    st. write(chain.invoke({"questions":input_text}))