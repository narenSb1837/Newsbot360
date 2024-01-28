import streamlit as st
from chtbot import chtreply
from datetime import date
from webscrape import *
import os
st.set_page_config(
    page_title="NewsBot360",
    page_icon=":newspaper:",
    layout="wide"
)
st.title("NewsBot360")

today = str(date.today())+'.txt'
print(today)
files = os.listdir('files')
if today not in files:
    with st.spinner("Scraping data..."):
            scrape('files/'+today)    

            st.success()

    
    
    
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Message NewsBot360..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chtreply(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        st.markdown(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})