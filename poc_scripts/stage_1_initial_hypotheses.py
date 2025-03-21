import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

import pandas as pd

from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta
)

from openai.types.beta.threads.text_delta_block import TextDeltaBlock

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# models = client.models.list()

# for model in models:
#     print(model.id)

st.set_page_config(page_title="Research Assistant", 
                   page_icon="🧠", 
                   layout="centered")
st.subheader("**Step 1: Define Clear Research Questions and Hypotheses**")
st.markdown("""
Begin by refining your initial ideas into testable hypotheses.

**Clarify your ecological questions:**
- What specifically are you trying to understand? (e.g., species interactions, community dynamics, impacts of environmental variables)

**Formulate hypotheses clearly as:**
- A clear statement predicting relationships among variables (e.g., "Plant biomass will decrease with increasing herbivore density.")

**Think about:**
- Ecological theory to guide expectations.
- Existing literature to contextualize your hypotheses.

**Output of this step:**
- Clearly written research questions.
- Explicit testable hypotheses.
""")


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "items": [
                                      {"type": "text",
                                       "content": "Lets work on your research questions first."}
                                  ]}]

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "files" not in st.session_state:
    st.session_state.files = []

if "data_summary" not in st.session_state:
    st.session_state.data_summary = []

if "file_id" not in st.session_state:
    st.session_state.file_id = []

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        
        st.session_state.file_uploaded = True
        st.session_state["files"].append(uploaded_file)

        # Create a thread for the user whith the uploaded file
        if "thread_id" not in st.session_state:
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
            print(st.session_state.thread_id)

    if button := st.button("Inspect the data"):

        if not st.session_state.file_uploaded:
            st.warning("Please upload a file first.")
            st.stop()

        else:
            # Create an agent that reads the data and returns its summary
            # Details of the summary can be later used to generate hypotheses

            with st.spinner("Inspecting the data ..."):

                for file in st.session_state.files:
                    oai_file = client.files.create(
                        file=file,
                        purpose="assistants"
                    )
                    st.session_state.file_id.append(oai_file.id)

                # Not sure whether this part is necessary
                message = client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content="Inspect the data"
                )

                data_summary_assistant = client.beta.assistants.create(
                    name="Data Summarizing Assistant",
                    instructions="Summarize the data. Provide column names, data types, and basic statistics.",
                    tools=[{"type": "code_interpreter"}],
                    model="gpt-4o"
                )

                client.beta.threads.update(
                    thread_id=st.session_state.thread_id,
                    tool_resources={"code_interpreter": {
                        "file_ids": [file_id for file_id in st.session_state.file_id]}
                    }
                )

                with st.chat_message("assistant"):
                    stream = client.beta.threads.runs.create(
                        thread_id=st.session_state.thread_id,
                        assistant_id=data_summary_assistant.id,
                        tool_choice={"type": "code_interpreter"},
                        stream=True
                    )

                    assistant_output = []
                    
                    for event in stream:
                        if isinstance(event, ThreadMessageCreated):
                            assistant_output.append({"type": "text",
                                                "content": ""})
                            assistant_text_box = st.empty()

                        elif isinstance(event, ThreadMessageDelta):
                            if isinstance(event.data.delta.content[0], TextDeltaBlock):
                                assistant_text_box.empty()
                                assistant_output[-1]["content"] += event.data.delta.content[0].text.value
                                assistant_text_box.markdown(assistant_output[-1]["content"])

                    st.session_state.data_summary.append(assistant_output)
                    print(f"The saved data summary: {st.session_state.data_summary}.")

            st.success("Done!")

            st.session_state.messages.append({"role": "assistant",
                                            "items": [
                                                {"type": "text",
                                                "content": "Great! Let's move on to hypotheses."}
                                            ]})
        

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        for item in message["items"]:
            if item["type"] == "text":
                st.write(item["content"])
            elif item["type"] == "file":
                # st.write(item["content"])
                dataframe = pd.read_csv(item["file"])
                st.write(dataframe)

if prompt := st.chat_input("Paste your hypotheses here, or upload your files (csv, txt)",
                           accept_file="multiple"):
    
    print(prompt)

    if prompt.text:
        st.session_state.messages.append({"role": "user",
                                          "items": [
                                              {"type": "text",
                                               "content": prompt.text}
                                          ]})
    if prompt.files:
        for file in prompt.files:
            st.session_state.files.append(file)
            st.session_state.messages.append({"role": "user",
                                              "items": [
                                                  {"type": "file",
                                                   "content": file.name,
                                                   "file": file}
                                              ]})