import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# https://tsjohnnychan.medium.com/a-chatgpt-app-with-streamlit-advanced-version-32b4d4a993fb

load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=openai_api_key)

# The developer message is a prompt that is prepended to the user's input
developer_filename = "prompts/plan-generation-developer-message.txt"
with open(developer_filename, "r", encoding="utf-8") as file:
    DEVELOPER_MESSAGE = file.read()


assistant = client.beta.assistants.create(
    name="Research Assistant",
    instructions="Write code to help with research tasks.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o-mini",
)

# Create a thread

# STREAMLIT APP
st.write("# Research Assistant Chat")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
# Initialize session state storage to store chat history and files
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# Using object notation
# add_selectbox = st.sidebar.selectbox(
#     "Select LLM model.",
#     ("4o", "o1", "o3-mini")
# )

# # Using "with" notation
# with st.sidebar:
#     add_radio = st.radio(
#         "Chose a work mode.",
#         ("Refinig hypotheses", "Running analyses", "Writing results")
#     )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Upload your data and hypotheses so we can start working on your analyses.", accept_file="multiple", 
                           file_type=["csv", "txt"]):
    
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt.text})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt.text)
    # If files are uploaded
    if prompt.files:
        for file in prompt.files:
            st.session_state["uploaded_files"].append(file)
            st.session_state["messages"].append({"role": "assistant", "content": f"Received file: {file.name}"})
            with st.chat_message("assistant"):
                st.write(f"Received file: {file.name}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {
                    "role": "developer", 
                    "content": DEVELOPER_MESSAGE
                    },
                *({
                    "role": m["role"], 
                    "content": m["content"]
                }
                for m in st.session_state.messages)
            ],
            stream=True
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.write("## Uploaded Files")
        if st.session_state["uploaded_files"]:
            for i, file_obj in enumerate(st.session_state["uploaded_files"], start=1):
                st.write(f"{i}. {file_obj.name}")
        else:
            st.write("No files uploaded yet.")