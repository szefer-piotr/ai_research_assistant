import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import pandas as pd
import pprint
import base64

from poc_scripts.assistant_event_handlers import EventHandler

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

refine_hypotheses_instructions = """
You are an expertin ecoogical studies that knows the literature in-and-out.
Task: Refine the provided hypotheses further based on the daset description and refined hypotheses.
The hypotheses should be more specific and actionable.
Always search the web to look for additional information that can help refine the hypotheses.
From your web search always provide a short summary of the information you found.
Provide your thought process and refine the hypotheses step by step.
The output should be a list of refined hypotheses.
"""

data_summary_assistant = client.beta.assistants.create(
    name="Data Summarizing Assistant",
    temperature=0,
    instructions=data_summary_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
)

def main():
    # Title
    st.markdown(
        """
        <h1 style='text-align: center; font-family: monospace; letter-spacing: 2px;'>
        RESEARCH ASSISTANT
        </h1>
        """,
        unsafe_allow_html=True
    )

    # -- Initialize session state --
    if "csv_file" not in st.session_state:
        st.session_state.csv_file = {}
    if "file_ids" not in st.session_state:
        st.session_state.file_ids = []
    if "hypotheses" not in st.session_state:
        st.session_state.hypotheses = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # We'll store which phase we're in: "upload", "processing_data", "data_summarized", "refining"
    if "phase" not in st.session_state:
        st.session_state.phase = "upload"

    # =============== PHASE: UPLOAD ===============
    if st.session_state.phase == "upload":
        st.sidebar.header("Upload Your Files")

        csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
        txt_file = st.sidebar.file_uploader("Upload TXT File", type=["txt"])

        st.subheader("File Overviews")

        # Show CSV Overview
        if csv_file is not None:
            st.write("#### CSV File Overview")
            try:
                df = pd.read_csv(csv_file)
                st.write(f"**Filename:** {csv_file.name}")
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
                st.dataframe(df.head())
                st.session_state.csv_file[csv_file.name] = csv_file
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

        # Show TXT Overview
        if txt_file is not None:
            st.write("#### TXT File Overview")
            try:
                text_content = txt_file.read().decode("utf-8")
                st.write(f"**Filename:** {txt_file.name}")
                st.write("**Content (first 200 chars):**")
                st.write(text_content[:200] + ("..." if len(text_content) > 200 else ""))
                st.session_state.hypotheses = text_content
            except Exception as e:
                st.error(f"Error reading TXT file: {e}")

        # Show "Process Data" button if both files are present
        if csv_file is not None and txt_file is not None:
            # Make the button green
            st.markdown(
                """
                <style>
                div.stButton > button:first-child {
                    background-color: green !important;
                    color: white;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            if st.sidebar.button("Process Data"):
                # We switch to the data-processing phase
                # Upload the CSV to OpenAI
                for file in st.session_state.csv_file:
                    openai_file = client.files.create(
                        file=st.session_state.csv_file[file],
                        purpose="assistants"
                    )
                    st.session_state.file_ids.append(openai_file.id)

                # Create the conversation thread that will do the data summary
                thread = client.beta.threads.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Refine the hypotheses: {st.session_state.hypotheses} using the attached dataset. List the refined hypotheses at the end.",
                            "attachments": [
                                {
                                    "file_id": st.session_state.file_ids[0],
                                    "tools": [{"type": "code_interpreter"}]
                                }
                            ]
                        }
                    ]
                )
                st.session_state.thread_id = thread.id
                st.session_state.phase = "processing_data"
                st.rerun()

    # =============== PHASE: PROCESSING_DATA (Data Summary Step) ===============
    elif st.session_state.phase == "processing_data":
        st.subheader("Data Summary Streaming")

        # We'll create the stream from the data_summary_assistant
        with st.chat_message("assistant"):
            stream = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=data_summary_assistant.id,
                tool_choice={"type": "code_interpreter"},
                stream=True
            )
            assistant_output = []
            code_input_expander = None
            code_input_block = None
            assistant_text_box = None

            for event in stream:
                if isinstance(event, ThreadRunStepCreated):
                    # Code is being written
                    if event.data.step_details.type == "tool_calls":
                        assistant_output.append({
                            "type": "code_input",
                            "content": ""
                        })
                        code_input_expander = st.status("Writing code ⏳ ...", expanded=True)
                        code_input_block = code_input_expander.empty()

                elif isinstance(event, ThreadRunStepDelta):
                    # Possibly partial code
                    if event.data.delta.step_details.tool_calls:
                        ci = event.data.delta.step_details.tool_calls[0].code_interpreter
                        if ci is not None and ci.input is not None and ci.input != "":
                            assistant_output[-1]["content"] += ci.input
                            code_input_block.empty()
                            code_input_block.code(assistant_output[-1]["content"])

                elif isinstance(event, ThreadRunStepCompleted):
                    # Code interpreter call is completed
                    if isinstance(event.data.step_details, ToolCallsStepDetails):
                        ci = event.data.step_details.tool_calls[0].code_interpreter
                        if ci.outputs and len(ci.outputs) > 0:
                            code_input_expander.update(label="Code", state="complete", expanded=False)
                            code_interpretor_outputs = ci.outputs[0]

                            # If we got images
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

                                    # Convert to base64
                                    with open(f"images/{image_file_id}.png", "rb") as f_:
                                        contents = f_.read()
                                        data_url = base64.b64encode(contents).decode("utf-8")

                                    # Display
                                    image_html = f'<p align="center"><img src="data:image/png;base64,{data_url}" width=600></p>'
                                    st.html(image_html)
                                    image_html_list.append(image_html)

                                assistant_output.append({
                                    "type": "image",
                                    "content": image_html_list
                                })

                            elif isinstance(code_interpretor_outputs, CodeInterpreterOutputLogs):
                                # Console logs
                                assistant_output.append({
                                    "type": "code_output",
                                    "content": ""
                                })
                                code_output = code_interpretor_outputs.logs
                                with st.status("Results", state="complete"):
                                    st.code(code_output)
                                    assistant_output[-1]["content"] = code_output
                        else:
                            # No outputs
                            code_input_expander.update(label="Code", state="complete", expanded=False)

                elif isinstance(event, ThreadMessageCreated):
                    # We have a new text message from the assistant
                    assistant_output.append({"type": "text", "content": ""})
                    assistant_text_box = st.empty()

                elif isinstance(event, ThreadMessageDelta):
                    # We got partial text
                    if isinstance(event.data.delta.content[0], TextDeltaBlock):
                        if assistant_text_box is not None:
                            assistant_text_box.empty()
                            assistant_output[-1]["content"] += event.data.delta.content[0].text.value
                            assistant_text_box.markdown(assistant_output[-1]["content"])

            # Store the final data summary message in st.session_state
            st.session_state.messages.append({"role": "assistant", "items": assistant_output})

        # Now that the data summary is done, show a button to move on
        if st.button("Done with Data Processing"):
            st.session_state.phase = "data_summarized"
            st.rerun()

    # =============== PHASE: DATA_SUMMARIZED (Show a button to refine) ===============
    elif st.session_state.phase == "data_summarized":
        st.markdown("## Data Processing Complete")
        st.markdown("Click the button below to refine hypotheses with LLM.")

        # Show the user's data summary if you'd like (optional). For now, let's skip it
        # or you can do something like:
        # for msg in st.session_state.messages:
        #     if msg["role"] == "assistant":
        #         st.write(msg["items"])

        # A green button to refine hypotheses
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: green !important;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("Refine hypotheses with LLM."):
            st.session_state.phase = "refining"
            st.rerun()

    # =============== PHASE: REFINING (Second LLM call) ===============
    elif st.session_state.phase == "refining":
        st.markdown("## Refining Hypotheses with LLM")

        # We'll do the second streaming call here
        # 1) Extract the 'text' items from the last data summary
        if len(st.session_state.messages) == 0:
            st.write("No data to refine!")
            return

        # We assume the last message is the data summary. Extract any text items:
        last_message = st.session_state.messages[-1]
        text_contents = [
            item["content"]
            for item in last_message["items"]
            if item["type"] == "text"
        ]
        joined_text = "\n".join(text_contents)

        chat_container = st.chat_message("assistant")
        partial_response_placeholder = chat_container.empty()
        partial_text = ""

        # Create the streaming response
        response = client.responses.create(
            model="gpt-4o",
            input=joined_text,
            instructions=refine_hypotheses_instructions,
            tools=[{"type": "web_search_preview"}],
            stream=True,
        )

        for event in response:
            if event.type == "response.text.delta":
                partial_text += event.delta
                partial_response_placeholder.markdown(partial_text)
            elif event.type == "response.completed":
                st.write("Done streaming!")
                break

        st.write("Refinement complete!")
        # If you want to store the final refined text in session_state, do so:
        st.session_state.messages.append({
            "role": "assistant",
            "items": [{
                "type": "text",
                "content": partial_text
            }]
        })
        # If you'd like a final "Done" button or something:
        if st.button("Done"):
            st.write("Thanks for refining!")
            # Optionally go to another phase or keep "refining" as final.

if __name__ == "__main__":
    main()
