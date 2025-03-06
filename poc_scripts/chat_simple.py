import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# https://tsjohnnychan.medium.com/a-chatgpt-app-with-streamlit-advanced-version-32b4d4a993fb

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


# st.title("🦜🔗 Quickstart App")


# def generate_response(input_text):
#     model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
#     st.info(model.invoke(input_text))


# with st.form("my_form"):
#     text = st.text_area(
#         "Enter text:",
#         "What are the three key pieces of advice for learning how to code?",
#     )
#     submitted = st.form_submit_button("Submit")
#     if not openai_api_key.startswith("sk-"):
#         st.warning("Please enter your OpenAI API key!", icon="⚠")
#     if submitted and openai_api_key.startswith("sk-"):
#         generate_response(text)