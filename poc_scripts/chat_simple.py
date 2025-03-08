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

executor_filename = "prompts/plan-generation-executor.txt"
with open(executor_filename, "r", encoding="utf-8") as file:
    EXECUTOR_MESSAGE = file.read()


assistant = client.beta.assistants.create(
    name="Research Assistant",
    instructions=EXECUTOR_MESSAGE,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)

# https://github.com/gabrielchua/dave

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
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state["thread_id"] = thread.id
    print(st.session_state["thread_id"])




# Select a GPT model
# add_selectbox = st.sidebar.selectbox(
#     "Select LLM model.",
#     ("4o", "o1", "o3-mini")
# )

# Add work mode
# with st.sidebar:
#     add_radio = st.radio(
#         "Chose a work mode.",
#         ("Refinig hypotheses", "Running analyses", "Writing results")
#     )

# Display previous messages
for message in st.session_state["messages"]:
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

    # Add user messages to the thread?
    message = client.beta.threads.messages.create(
        thread_id=st.session_state["thread_id"],
        role="user",
        content=prompt.text
    )

    # If files are uploaded
    if prompt.files:
        for file in prompt.files:
            # Append the file to the session state.
            st.session_state["uploaded_files"].append(file)
            # Display a message that a file was recieved.
            st.session_state["messages"].append({"role": "assistant", "content": f"Received file: {file.name}"})
            
            # Initiate the file id stroage in session state
            st.session_state["file_id"] = []

            # Process each file, add files to openai and store the file id in the session state
            for file in st.session_state["uploaded_files"]:
                openai_file = client.files.create(
                    file=file,
                    purpose='assistants'
                )
                st.session_state["file_id"].append(openai_file.id)
                print(f"Uploaded new file: \t {openai_file.id}")

            with st.chat_message("assistant"):
                print(st.session_state["uploaded_files"][0])
                st.write(f"Received file: {file.name}")

        # Update the assitants' thread with uploaded file
        client.beta.threads.update(
            thread_id=st.session_state["thread_id"],
            tool_resources={"code_interpreter": {"file_ids": [file_id for file_id in st.session_state.file_id]}
                            }
            )
        print(f"[INFO] {client.beta.threads}")

        # How about the message history???

    # Display downloaded files
    with st.sidebar:
        st.write("## Uploaded Files")
        if st.session_state["uploaded_files"]:
            for i, file_obj in enumerate(st.session_state["uploaded_files"], start=1):
                st.write(f"{i}. {file_obj.name}")
        else:
            st.write("No files uploaded yet.")


    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        # Create a run for the assistant. Assistant will optionally use a tool?
        # stream = client.beta.threads.runs.create(
        #     thread_id=st.session_state["thread_id"],
        #     assistant_id=assistant.id,
        #     tool_choice={"type": "code_interpreter"},
        #     stream=True
        # )
        
        # Previous stream
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