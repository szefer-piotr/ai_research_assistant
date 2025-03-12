import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

from prompts import DEVELOPER_MESSAGE, EXECUTOR_MESSAGE
# from assistants import assistant
from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta
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

        with st.chat_message("user"):
            st.markdown(prompt.text)

        with st.chat_message("assistant"):
            stream = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant.id,
                tool_choice={"type": "code_interpreter"},
                stream=True
            )

            assistant_output = []

            # for event in stream:
            #     print(event)
            #     if isinstance(event, ThreadRunStepCreated):
            #         if event.delta.step_details.type == "tool_calls":
            #             assistant_output.append(
            #                 {
            #                     "type": "code_input",
            #                     "content": ""
            #                 }
            #             )
            #             code_input_expander = st.status("Writing code ...", expanded=True)
            #             code_input_block = code_input_expander.empty()