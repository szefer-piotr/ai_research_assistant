import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from prompts import DEVELOPER_MESSAGE, EXECUTOR_MESSAGE
# from assistants import assistant
from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta
)

from openai.types.beta.threads.text_delta_block import TextDeltaBlock
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs
    )


load_dotenv()

#UI

st.set_page_config(page_title="Research Assistant",
                    page_icon="🕵️")

st.subheader("🔮 Research Assistant")

# Upload box
uploaded_file = st.sidebar.file_uploader("**Upload your dataset(s) and hypotheses...**", type=["csv", "xlsx"])
upload_btn = st.sidebar.button("Upload")


# Initiate session states
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                        "items": [
                                            {"type": "text", 
                                            "content": "Upload your files and lets discuss what you would like to work on today?"
                                            }]}]

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

if "files" not in st.session_state:
    st.session_state.files = []

if "file_id" not in st.session_state:
    st.session_state.file_id = []


# print("***" * 20)
# print(f"[INFO] What is up with the {uploaded_file}")
# print("***" * 20)

# Define an OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=openai_api_key)

if (uploaded_file) == None and not (st.session_state.file_uploaded):
    # st.write("Upload your files.")
    if upload_btn:
        with st.sidebar:
            st.warning("No files were selected!")
            time.sleep(1)
    # st.rerun()

    # Allow for a conversation without having any files
    if prompt := st.chat_input("Ask a question or upload your data..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            {"role": "assistant",
             "items": [
                 {"type": "text", 
                  "content": prompt
                  }
                  ]
            }
        )

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )
            st.write_stream(stream)





elif uploaded_file is not None:

    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions=EXECUTOR_MESSAGE,
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o",
    )

    print(f"Created an assistant with an ID: {assistant.id}")

    # Initialise session state variables
    # That does anything by itself. Initializes a text box and a button.
    # text_box = st.empty()
    # qn_btn = st.empty()

    if upload_btn:
        # Check if the file is already in the uploaded list
        if uploaded_file.name in [file.name for file in st.session_state["files"]]:
            with st.sidebar:
                st.warning("File has already been uploaded!")

        else:
            st.toast("File(s) uploaded successfully")
            st.session_state["files"].append(uploaded_file)
            st.session_state["file_uploaded"] = True
        
            # This part may have to be started when user specifies which file to use.
            for file in st.session_state["files"]:
                oai_file = client.files.create(
                    file=file,
                    purpose="assistants"
                )
            
                st.session_state["file_id"].append(oai_file.id)
                print(f"Uploaded new file: \t {oai_file.id}")

           
if st.session_state["file_uploaded"]:
    print(f"FILE UPLADED")

    # with st.sidebar:
    #     st.header("File List")
    #     for idx, file in enumerate(st.session_state["files"], start=1):
    #         print(idx, file)
    #         st.write(f"{idx}. {file.name}")

    with st.sidebar:
        st.header("📂 Files")

        if "files" in st.session_state:
            files = st.session_state["files"]
            if len(files) == 0:
                st.write("No files available.")
            else:
                for file in files:
                    col1, col2 = st.sidebar.columns([0.8, 0.2])
                    col1.write(file.name)
                    # The "x" button to delete files
                    if col2.button("❌", key=f"delete_{file.name}"):
                        st.session_state["files"].remove(file)
                        st.rerun()

        else:
            st.write("Files not found in session state.")

    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
        print(st.session_state.thread_id)
    
    client.beta.threads.update(
        thread_id=st.session_state.thread_id,
        tool_resources={"code_interpreter": {"file_ids": [file_id for file_id in st.session_state.file_id]}}
    )

    #UI
    for message in st.session_state.messages:
        print(f"The message: {message}")
        with st.chat_message(message["role"]):
            for item in message["items"]:
                item_type = item["type"]
                if item_type == "text":
                    st.markdown(item["content"])
                elif item_type == "image":
                    for image in item["content"]:
                        st.image(image)
                elif item_type == "code_input":
                    with st.status("Code", state="complete"):
                        st.code(item["content"])
                elif item_type == "code_output":
                    with st.status("Results", state="complete"):
                        st.code(item["content"])

    if prompt := st.chat_input(
        "Ask a question or upload your hypotheses...", 
        accept_file="multiple", 
        file_type=["csv", "txt"]
        ):

        st.session_state.messages.append(
            {
                "role": "user",
                "items": [
                    {"type": "text",
                    "content": prompt.text}
                ]
            }    
        )

        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt.text
        )

        # Chat message container for the user
        with st.chat_message("user"):
            st.markdown(prompt.text)

        # Chat message container for the assistant
        with st.chat_message("assistant"):
            # First create an assistant
            stream = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant.id,
                tool_choice={"type": "code_interpreter"},
                stream=True
            )

            # Initialize a list of outputs
            assistant_output = []
            
            # Handle events in the stream
            for event in stream:
                # print(f"[INFO] Event:\n {type(event)}")
                # if event == thread.run.step.delta:
                if isinstance(event, ThreadRunStepCreated):
                    if event.data.step_details.type == "tool_calls":
                        assistant_output.append({"type": "code_input",
                                                "content": ""})
                        
                        code_input_expander = st.status("Writing code ...", expanded=True)
                        code_input_block = code_input_expander.empty()

                if isinstance(event, ThreadRunStepDelta):
                    if event.data.delta.step_details.tool_calls[0].code_interpreter is not None:
                        code_interpreter = event.data.delta.step_details.tool_calls[0].code_interpreter
                        code_input_delta = code_interpreter.input
                        # print(f"[INFO] Code input delta: {code_input_delta}")
                        if (code_input_delta is not None) and (code_input_delta != ""):
                            assistant_output[-1]["content"] += code_input_delta
                            code_input_block.empty()
                            code_input_block.code(assistant_output[-1]["content"])
                            # This part is added so that the 
                        # code_input_expander.update(label="Code", state="complete", expanded=False)

                elif isinstance(event, ThreadRunStepCompleted):
                    if isinstance(event.data.step_details, ToolCallsStepDetails):
                        code_interpreter = event.data.step_details.tool_calls[0].code_interpreter
                        code_input_expander.update(label="Code", state="complete", expanded=False)

                        for output in code_interpreter.outputs:
                            image_data_bytes_list = []
                            
                            if isinstance(output, CodeInterpreterOutputImage):
                                image_file_id = output.image.file_id                                   
                                image_data = client.files.content(image_file_id)
                                image_data_bytes = image_data.read()
                                st.image(image_data_bytes)
                                image_data_bytes_list.append(image_data_bytes)

                            if isinstance(output, CodeInterpreterOutputLogs):
                                # print(f"[INFO] This is a log. Show it in the code window.")
                                assistant_output.append({"type": "code_input",
                                                        "content": ""})
                                code_output = output.logs
                                with st.status("Results", state="complete"):
                                    st.code(code_output)
                                    assistant_output[-1]["content"] = code_output

                            assistant_output.append({
                                "type": "image",
                                "content":image_data_bytes_list
                            })
                
                elif isinstance(event, ThreadMessageCreated):
                    assistant_output.append({"type": "text",
                                            "content": ""})
                    assistant_text_box = st.empty()

                elif isinstance(event, ThreadMessageDelta):
                    if isinstance(event.data.delta.content[0], TextDeltaBlock):
                        assistant_text_box.empty()
                        assistant_output[-1]["content"] += event.data.delta.content[0].text.value
                        assistant_text_box.markdown(assistant_output[-1]["content"])

            st.session_state.messages.append({"role": "assistant", "items": assistant_output})