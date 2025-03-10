import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

from prompts import DEVELOPER_MESSAGE, EXECUTOR_MESSAGE
# from assistants import assistant

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=openai_api_key)

# Create an assistant
client = OpenAI(api_key=openai_api_key)

assistant = client.beta.assistants.create(
    name="Research Assistant",
    instructions=EXECUTOR_MESSAGE,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)

print(f"Created an assistant with an ID: {assistant.id}")

st.set_page_config(page_title="Research Assistant",
                   page_icon="🕵️")

# Initialise session state variables

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What you would like to work on today?"}]

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "assistant_text" not in st.session_state:
    st.session_state.assistant_text = [""]

if "code_input" not in st.session_state:
    st.session_state.code_input = []

if "code_output" not in st.session_state:
    st.session_state.code_output = []

if "disabled" not in st.session_state:
    st.session_state.disabled = False

st.subheader("Research Assistant Chat")
st.markdown("Upload your data and start your analyses!")

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# That does anything by itself. Initializes a text box and a button.
text_box = st.empty()
qn_btn = st.empty()

if prompt := st.chat_input(
    "Upload csv or txt...", 
    accept_file="multiple", 
    file_type=["csv", "txt"]
    ):

    # Create a new thread and add its id to the session state
    # Create a new thread
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
        print(st.session_state.thread_id)

    # Prompt can have text or it can be a file.

    # If it is text add it to the

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "contente": prompt})

# question = text_box.text_area("Ask a question", disabled=st.session_state.disabled)