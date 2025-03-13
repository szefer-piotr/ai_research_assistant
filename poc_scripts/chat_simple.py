import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing_extensions import override
from openai import AssistantEventHandler
# from utils import EventHandler

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

class EventHandler(AssistantEventHandler):    
  @override
  def on_text_created(self, text) -> None:
    st.session_state["messages"].append({"role": "assistant", "content": text})
    print(f"\nassistant > ", end="", flush=True)
      
  @override
  def on_text_delta(self, delta, snapshot):
    st.session_state["messages"].append({"role": "assistant", "content": text})
    print(delta.value, end="", flush=True)
      
  def on_tool_call_created(self, tool_call):
    st.session_state["messages"].append({"role": "assistant", "content": tool_call})
    print(f"\nassistant > {tool_call.type}\n", flush=True)
    
  
  def on_tool_call_delta(self, delta, snapshot):
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        st.session_state["messages"].append({"role": "assistant", "content": delta.code_interpreter.input})
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        st.session_state["messages"].append({"role": "assistant", "content": delta.code_interpreter.outputs})
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)

# STREAMLIT APP
st.write("# Research Assistant Chat")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
# Initialize session state storage to store chat history and files
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What you would like to work on today?"}]
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state["thread_id"] = thread.id
    print(f"Created a thread: {st.session_state['thread_id']}")

# Should I add client.beta.threads.messages.create here as well? How should I update it?

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
    
    print("[INFO] ... ")

    # Add user messages to the thread?
    if (text := prompt.text):
        message = client.beta.threads.messages.create(
            thread_id=st.session_state["thread_id"],
            role="user",
            content=text
        )
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt.text)

        # Previous stream
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
                    for m in st.session_state["messages"])
                ],
                stream=True
            )
            
            response = st.write_stream(stream)

    # If files are uploaded
    if prompt.files:
        for file in prompt.files:
            # Append the file to the session state.
            st.session_state["uploaded_files"].append(file)
            
            # Display a message that a file was recieved.
            # print("******************************SHIT")
            st.session_state["messages"].append({"role": "user", "content": f"I have uploaded a file: {file.name}"})
            
            # Initiate the file id storage in session state
            st.session_state["file_id"] = []
            
            print("[INFO] Load files into the session state  ")

            # Process each file, add files to openai and store the file id in the session state
            openai_file = client.files.create(
                file=file,
                purpose='assistants'
                )
            
            st.session_state["file_id"].append(openai_file.id)
            
            with st.chat_message("user"):
                st.write(f"I have uploaded a file: {file.name}")

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

            with client.beta.threads.runs.stream(
                thread_id=st.session_state["thread_id"],
                assistant_id=assistant.id,
                instructions=EXECUTOR_MESSAGE,
                event_handler=EventHandler(),
            ) as stream:
                st.write_stream(stream.text_deltas) # This writes all events into the console
                stream.until_done()
            
            #----------------------------------------------------------------------
            # # Create a run for the assistant. Assistant will optionally use a tool?
            # stream = client.beta.threads.runs.create(
            #     thread_id=st.session_state["thread_id"],
            #     assistant_id=assistant.id,
            #     tool_choice={"type": "code_interpreter"},
            #     stream=True
            # )

            # # Build an event handler.
            response = st.write_stream(stream)
            #----------------------------------------------------------------------
            st.session_state.messages.append({"role": "assistant", "content": response})