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


# That does anything by itself. Initializes a text box and a button.
# text_box = st.empty()
# qn_btn = st.empty()


#UI
st.subheader("🔮 Research Assistant")

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

file_upload_box = st.empty()
upload_btn = st.empty()

if not st.session_state["file_uploaded"]:
    st.session_state["files"] = file_upload_box.file_uploader(
        "Upload your dataset(s) and hypotheses...",
        accept_multiple_files=True,
        type=['csv','txt']
    )

    if upload_btn.button("Upload"):
        st.session_state["file_id"] = []
        for file in st.session_state["files"]:
            oai_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            st.session_state["file_id"].append(oai_file.id)
            print(f"Uploaded new file: \t {oai_file.id}")
        st.toast("File(s) uploaded successfully", icon=":rocket:")
        st.session_state["file_uploaded"] = True
        file_upload_box.empty()
        st.rerun()

        
if st.session_state["file_uploaded"]:
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
        print(st.session_state.thread_id)
    
    client.beta.threads.update(
        thread_id=st.session_state.thread_id,
        tool_resources=[{"code_interpreter": {"file_id" for file_id in st.session_state.file_id}}]
    )

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    #UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for item in message["items"]:
                item_type = item["type"]

if prompt := st.chat_input(
    "Upload csv or txt...", 
    accept_file="multiple", 
    file_type=["csv", "txt"]
    ):

    # Create a new thread and add its id to the session state
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
        print(st.session_state.thread_id)

    if prompt.text:
        print(prompt.text)


    if prompt.files:
        # Because we have multiple files possibly uploaded we loop throug each one
        for file in prompt.files:
            st.session_state.messages.append({"role": "user", "content": f"I have uploaded a file: {file.name}"})
            # Add file uploader
            openai_file = client.files.create(
                    file=file,
                    purpose='assistants'
                    )
            # Update a thread with each file
            client.beta.threads.update(
                thread_id=st.session_state.thread_id,
                tool_resources={"code_interpreter": {"file_ids": openai_file.id}}
                )
    # Prompt can have text or it can be a file.
    # if prompt.text

    # If it is text add it to the

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# question = text_box.text_area("Ask a question", disabled=st.session_state.disabled)