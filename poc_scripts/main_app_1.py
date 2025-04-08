import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import pandas as pd

import pprint

import base64

from assistant_event_handlers import EventHandler

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
client = OpenAI(api_key=openai_api_key)


data_summary_instructions = """
Task: Summarize the provided dataset by analyzing its columns.
Extract and list all column names.
For each column:
Infer a human-readable description of what the column likely represents.
Identify the data type (e.g., categorical, numeric, text, date).
Count the number of unique values.
Output Format: Return a Python dictionary where each key is the column name, and each value is another dictionary with the following structure:
{
  "column_name": <<name of the analyzed column>>,
  "description": <<inferred description of the column>>,
  "type": <<inferred data type>>,
  "unique_value_count": <<number of unique values>>
}
The dictionary should contain one entry per column in the dataset.
"""

data_summary_assistant = client.beta.assistants.create(
    name="Data Summarizing Assistant",
    temperature=0,
    instructions=data_summary_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
)


def main():
    st.title("CSV & TXT File Uploader")

    # Initialize session state for files
    if "csv_file" not in st.session_state:
        st.session_state["csv_file"] = {}
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = {}
    if "file_ids" not in st.session_state:
        st.session_state["file_ids"] = []
    if "hypotheses" not in st.session_state:
        st.session_state["hypotheses"] = ""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    # This boolean flag controls whether we hide the file overview
    if "processing" not in st.session_state:
        st.session_state["processing"] = False

    # Only show the chat messages if we are already processing
    if st.session_state["processing"]:
        st.subheader("Streamed Messages")
        for message in st.session_state.messages:
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
        # Stop here so we don't show file overviews once "processing" is True.
        return

    # ----------- If not processing, show file overviews ------------
    st.sidebar.header("Upload Your Files")

    # Upload CSV
    csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    # Upload TXT
    txt_file = st.sidebar.file_uploader("Upload TXT File", type=["txt"])

    st.subheader("File Overviews")

    if csv_file is not None:
        st.write("#### CSV File Overview")
        try:
            df = pd.read_csv(csv_file)
            st.write(f"**Filename:** {csv_file.name}")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {list(df.columns)}")
            st.dataframe(df.head())
            st.session_state.csv_file[csv_file.name] = csv_file

            print(st.session_state.csv_file[csv_file.name])

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    if txt_file is not None:
        st.write("#### TXT File Overview")
        try:
            text_content = txt_file.read().decode("utf-8")
            st.write(f"**Filename:** {txt_file.name}")
            st.write("**Content (first 200 chars):**")
            st.write(text_content[:200] + ("..." if len(text_content) > 200 else ""))
            st.session_state.hypotheses = text_content

            print(st.session_state.hypotheses)

        except Exception as e:
            st.error(f"Error reading TXT file: {e}")

    # --- Once both CSV & TXT are present, show the "Process" button ---
    if csv_file is not None and txt_file is not None:
        button = st.sidebar.button("Process Hypotheses")
        if button:
            # Set "processing" so we hide file overviews on next rerun
            st.session_state["processing"] = True

            # Create files in OpenAI
            for file in st.session_state.csv_file:
                openai_file = client.files.create(
                    file=st.session_state.csv_file[file],
                    purpose="assistants"
                )
                st.session_state.file_ids.append(openai_file.id)

            # Create a thread for the whole session
            # Add files at the level of the thread
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Refine the hypotheses: {st.session_state.hypotheses} using the attached dataset.",
                        "attachments": [
                            {
                                "file_id": st.session_state.file_ids[0],
                                "tools": [{"type": "code_interpreter"}]
                            }
                        ]
                    }
                ]
            )
            print(thread)
            st.session_state.thread_id = thread.id

            with st.chat_message("assistant"):
                stream = client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=data_summary_assistant.id,
                    tool_choice={"type": "code_interpreter"},
                    stream=True
                )
                assistant_output = []

                for event in stream:
                    print(event)
                    if isinstance(event, ThreadRunStepCreated):
                        if event.data.step_details.type == "tool_calls":
                            assistant_output.append({
                                "type": "code_input", 
                                "content": ""
                            })
                            code_input_expander = st.status("Writing code ⏳ ...", expanded=True)
                            code_input_block = code_input_expander.empty()

                    elif isinstance(event, ThreadRunStepDelta):
                        if event.data.delta.step_details.tool_calls:
                            ci = event.data.delta.step_details.tool_calls[0].code_interpreter
                            if ci is not None and ci.input is not None and ci.input != "":
                                assistant_output[-1]["content"] += ci.input
                                code_input_block.empty()
                                code_input_block.code(assistant_output[-1]["content"])

                    elif isinstance(event, ThreadRunStepCompleted):
                        if isinstance(event.data.step_details, ToolCallsStepDetails):
                            ci = event.data.step_details.tool_calls[0].code_interpreter
                            # Safely handle empty or None outputs
                            if ci.outputs and len(ci.outputs) > 0:
                                code_input_expander.update(label="Code", state="complete", expanded=False)
                                code_interpretor_outputs = ci.outputs[0]

                                # Image
                                if isinstance(code_interpretor_outputs, CodeInterpreterOutputImage):
                                    image_html_list = []
                                    for output in ci.outputs:
                                        image_file_id = output.image.file_id
                                        image_data = client.files.content(image_file_id)

                                        # Save file
                                        image_data_bytes = image_data.read()
                                        os.makedirs("images", exist_ok=True)
                                        with open(f"images/{image_file_id}.png", "wb") as file:
                                            file.write(image_data_bytes)

                                        # Open file and encode as data
                                        file_ = open(f"images/{image_file_id}.png", "rb")
                                        contents = file_.read()
                                        data_url = base64.b64encode(contents).decode("utf-8")
                                        file_.close()

                                        # Display image
                                        image_html = f'<p align="center"><img src="data:image/png;base64,{data_url}" width=600></p>'
                                        st.html(image_html)

                                        image_html_list.append(image_html)

                                    assistant_output.append({
                                        "type": "image",
                                        "content": image_html_list
                                    })

                                # Console log
                                elif isinstance(code_interpretor_outputs, CodeInterpreterOutputLogs):
                                    assistant_output.append({
                                        "type": "code_output",
                                        "content": ""
                                    })
                                    code_output = code_interpretor_outputs.logs
                                    with st.status("Results", state="complete"):
                                        st.code(code_output)
                                        assistant_output[-1]["content"] = code_output
                            else:
                                # No outputs from code_interpreter
                                code_input_expander.update(label="Code", state="complete", expanded=False)

                    elif isinstance(event, ThreadMessageCreated):
                        assistant_output.append({"type": "text", "content": ""})
                        assistant_text_box = st.empty()

                    elif isinstance(event, ThreadMessageDelta):
                        if isinstance(event.data.delta.content[0], TextDeltaBlock):
                            assistant_text_box.empty()
                            assistant_output[-1]["content"] += event.data.delta.content[0].text.value
                            assistant_text_box.markdown(assistant_output[-1]["content"])

                st.session_state.messages.append({"role": "assistant", "items": assistant_output})
            # Rerun so that on the next load, "processing" is True and we show the streamed messages only
            st.rerun()


if __name__ == "__main__":
    main()
