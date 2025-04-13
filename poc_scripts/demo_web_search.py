import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


if 'response' not in st.session_state:
    st.session_state.response = None

st.title("Web Browsing Response")
st.markdown(""" Use the responses API """)

user_query = st.text_input("Enter your query:", placeholder="e.g. What are the latest news in ecology?")

show_raw = st.checkbox("Show raw response")

if st.button("Search"):
    if not user_query:
        st.warning("Please enter a query firts!")
    else:
        with st.spinner("Searching the web..."):
            try:
                st.session_state.response = client.responses.create(
                    model="gpt-4o",
                    tools=[{"type": "web_search_preview"}],
                    input=user_query
                )

            except Exception as e:
                st.error(f"An error occured: {str(e)}")
                st.info("Make sure you have set your OPENAI_API_KEY in the .env file")

if st.session_state.response:
    if show_raw:
        st.markdown(st.session_state.response)
    st.markdown(st.session_state.response.output_text)