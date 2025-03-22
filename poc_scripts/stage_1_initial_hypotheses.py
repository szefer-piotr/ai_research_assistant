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

# Keep the message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "items": [
                                      {"type": "text",
                                       "content": "Lets work on your research questions first."}
                                  ]}]

# Boolean indicator
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# Store UploadedFile objects
if "files" not in st.session_state:
    st.session_state.files = {}

# This is to store the data summary
if "data_summary" not in st.session_state:
    st.session_state.data_summary = []

if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = []

# Store the file IDs for the thread
if "file_id" not in st.session_state:
    st.session_state.file_id = []

# Store the thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = []

# Things happening in the sidebar
with st.sidebar:
    # If the file is not uploaded yet upload it, check if its name is already in the files
    if not st.session_state.file_uploaded:

        # Display the file uploader
        uploaded_file = st.file_uploader("Choose a file", type=['csv'])
        
        # Evaluate whether the file is uploaded
        if uploaded_file is not None:
            
            # Check if the file has already been uploaded
            if uploaded_file.name in st.session_state.files.keys():
                print(f"Uploaded file name: {uploaded_file.name}")
                st.warning("This file has already been uploaded.")
                st.stop()

            else: 
                # Display the uploaded file
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)

                # Save the uploaded file in a dictionary
                st.session_state.files[uploaded_file.name] = uploaded_file
                # Change the boolean indicator
                st.session_state.file_uploaded = True

                # Create a thread for the user whith the uploaded file
                thread = client.beta.threads.create()
                st.session_state.thread_id = thread.id
                
                print(f"\nThread created with an ID {st.session_state.thread_id}")
                
                # Rerun the script to move to the next step
                st.rerun()

    # # Sidebar functionality after file already uploaded
    else:
        selected_file = st.selectbox("Select a file to summarize/read", options=list(st.session_state.files.keys()))

        if selected_file:
            file_obj = st.session_state.files[selected_file]
            file_obj.seek(0)

            try:
                df = pd.read_csv(file_obj)
                st.subheader("Data preview:")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading the file: {e}")

        if button := st.button("Inspect the data"):

            with st.spinner("Inspecting the data ..."):

                for file in st.session_state.files:

                    print(f"This is the file in the sesssion_state files {file}")

                    # Create a file for the assistant
                    oai_file = client.files.create(
                        file=st.session_state.files[file],
                        purpose="assistants"
                    )
                    # Append the file ID to the list
                    st.session_state.file_id.append(oai_file.id)

                # Create an assistant for data summarization
                data_summary_assistant = client.beta.assistants.create(
                    name="Data Summarizing Assistant",
                    instructions="Summarize the data. Provide column names, data types, and basic statistics.",
                    tools=[{"type": "code_interpreter"}],
                    model="gpt-4o"
                )

                # Update the thread with the assistant
                client.beta.threads.update(
                    thread_id=st.session_state.thread_id,
                    tool_resources={"code_interpreter": {
                        "file_ids": [file_id for file_id in st.session_state.file_id]}
                    }
                )

                data_summary_box = st.empty()

                with data_summary_box:
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
                                                "content": "I have a summary of your dataset! We will use it to refine your hypotheses."}
                                            ]})
    
    # This is the summay of the data
    if st.session_state.data_summary:
        st.write(st.session_state.data_summary)
        

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

if prompt := st.chat_input("Paste your hypotheses here",
                           accept_file=True):
    
    print(prompt)

    if prompt.text:
        st.sesssion_state.hypotheses.append(prompt.text)
        # st.session_state.messages.append({"role": "user",
        #                                   "items": [
        #                                       {"type": "text",
        #                                        "content": prompt.text}
        #                                   ]})
    
    if prompt.files:
        file = prompt.files[0]
        file.seek(0)
        content = file.read().decode("utf-8")
        st.session_state.hypotheses.append(content)
        st.success("Hypotheses uploaded successfully!")

if st.session_state.hypotheses:
    st.subheader("Current Hypotheses")
    for idx, hypo in enumerate(st.session_state.hypotheses, start=1):
        st.markdown(f"**{idx}.** {hypo}")

# Optional: button to clear session state
if st.button("Clear all hypotheses"):
    st.session_state.hypotheses.clear()
    st.success("All hypotheses cleared.")