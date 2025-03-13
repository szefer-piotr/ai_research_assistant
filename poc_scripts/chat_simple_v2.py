import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

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
    st.session_state["messages"] = [{"role": "assistant",
                                        "items": [
                                            {"type": "text", 
                                            "content": "What you would like to work on today?"
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


# That does anything by itself. Initializes a text box and a button.
# text_box = st.empty()
# qn_btn = st.empty()


#UI
st.subheader("🔮 Research Assistant")

# Display previous messages
# for message in st.session_state["messages"]:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

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
        st.toast("File(s) uploaded successfully")
        st.session_state["file_uploaded"] = True
        file_upload_box.empty()
        st.rerun()

        
if st.session_state["file_uploaded"]:
    print(f"FILE UPLADED")
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
        print(st.session_state.thread_id)
    
    client.beta.threads.update(
        thread_id=st.session_state.thread_id,
        tool_resources={"code_interpreter": {"file_ids": [file_id for file_id in st.session_state.file_id]}}
    )

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

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
                        st.html(image)
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
                print(f"[INFO] Event:\n {type(event)}")
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
                        print(f"[INFO] Code input delta: {code_input_delta}")
                        if (code_input_delta is not None) and (code_input_delta != ""):
                            assistant_output[-1]["content"] += code_input_delta
                            code_input_block.empty()
                            code_input_block.code(assistant_output[-1]["content"])

                elif isinstance(event, ThreadRunStepCompleted):
                    if isinstance(event.data.step_details, ToolCallsStepDetails):
                        print(f"[INFO] Event data step details: {event.data.step_details.tool_calls}")
                        code_interpreter = event.data.step_details.tool_calls[0].code_interpreter
                        print(f"[INFO] Code interpreter:\n {code_interpreter.outputs}")
                        
                        if code_interpreter.outputs:
                            code_interpreter_outputs = code_interpreter.outputs[0]
                            print(f"[INFO] Code interpreter outputs:\n {code_interpreter_outputs}")
                            code_input_expander.update(label="Code", state="complete", expanded=False)
                            
                            # Image
                            if isinstance(code_interpreter_outputs, CodeInterpreterOutputImage):
                                image_html_list = []
                                for output in code_interpreter.outputs:
                                    image_file_id = output.image.file_id
                                    image_data = client.files.content(image_file_id)

                                    image_data_bytes = image_data.read()

                                    with open(f"images/{image_file_id}.png", "rb") as fle:
                                        file.write(image_data_bytes)
                                    
                                    file_ = open(f"images/{image_file_id}.png", "rb")
                                    contents = file_.read()
                                    data_url = base64.b64encode(contents).decode("utf-8")
                                    file_.close()

                                    # Display image
                                    image_html = f'<p align="center"><img src="data:image/png;base64,{data_url}" width=600></p>'
                                    st.html(image_html)

                                    image_html_list.append(image_html)
                                
                                assistant_output.append({"type": "image",
                                                         "content": image_html_list})
                                
                            elif isinstance(code_interpreter_outputs, CodeInterpreterOutputLogs):
                                assistant_output.append({"type": "code_input",
                                                         "content": ""})
                                code_output = code_interpreter.outputs[0].logs
                                with st.status("Results", state="complete"):
                                    st.code(code_output)
                                    assistant_output[-1]["content"] = code_output
                
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