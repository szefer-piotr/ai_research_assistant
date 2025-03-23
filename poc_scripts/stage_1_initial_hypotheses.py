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
    st.session_state.data_summary = [
        {
            "type":"text",
            "content":"The dataset consists of 61 columns. Here are the column names along with their data types:\n\n1. **INDEX_OF_INDIVIDUALS**: int\n2. **Bee.species**: object (string)\n3. **Species.code**: object (string)\n4. **Sex**: object (string)\n5. **Site.number**: object (string)\n6. **Year**: int\n7. **Month**: object (string)\n8. **Day**: int\n9. **Family**: object (string)\n10. **Social.behavior**: object (string)\n11. Additional columns related to impervious surface area and population density, all appearing as strings at first sight but potentially representing numerical values. These columns include:\n    - **Impervious.surface.area.in.buffer.250.m.[mean]**\n    - **Impervious.surface.area.in.buffer.500.m.[mean]**\n    - **Impervious.surface.area.in.buffer.750.m.[mean]**\n    - **Impervious.surface.area.in.buffer.1000.m.[mean]**\n    - **Impervious.surface.area.in.buffer.1500.m.[mean]**\n    - **Population.density.in.buffer.250.m**\n    - **Population.density.in.buffer.500.m**\n    - **Population.density.in.buffer.750.m**\n    - **Population.density.in.buffer.1000.m**\n    - **Population.density.in.buffer.1500.m**\n\nLet's convert any necessary columns to numeric data types and then provide basic statistics for the numerical columns."},
        {
            "type":"text",
            "content":"Here are the basic statistics for the numeric columns in the dataset:\n\n1. **INDEX_OF_INDIVIDUALS**: \n   - Mean: 2882\n   - Min: 1\n   - Max: 5763\n\n2. **Year**:\n   - Mean: 2018.62\n   - Min: 2018\n   - Max: 2019\n\n3. **Day**:\n   - Mean: 10.15\n   - Min: 1\n   - Max: 28\n\n4. **Lifespan (month)**:\n   - Mean: 4.79\n   - Min: 1\n   - Max: 8\n\n5. **Shortest Distance Between Sites (m)**:\n   - Mean: 4684.31\n   - Min: 527.72\n   - Max: 20068.52\n\n6. **Year of Research**:\n   - Mean: 2018.62\n   - Min: 2018\n   - Max: 2019\n\n7. **Coverage of Bee Food Plant Species (%)**:\n   - Mean: 41.77\n   - Min: 17\n   - Max: 65\n\n8. **Floral Richness**: \n   - Mean: 71.27\n   - Min: 32\n   - Max: 130\n\n9. **Alien Floral Richness (%)**:\n   - Mean: 30.19\n   - Min: 21.95\n   - Max: 39.71\n\n10. **Native Floral Richness (%)**:\n    - Mean: 54.03\n    - Min: 32.22\n    - Max: 75.76\n\n11. **Impervious Surface Area in Buffers (mean values)**:\n    - 250 m: Mean = 31.60, Min = 0.51, Max = 81.18\n    - 500 m: Mean = 29.33, Min = 0.20, Max = 73.53\n    - 750 m: Mean = 28.68, Min = 0.12, Max = 67.17\n    - 1000 m: Mean = 27.94, Min = 0.26, Max = 64.89\n    - 1500 m: Mean = 26.32, Min = 0.59, Max = 61.13\n\n12. **Population Density in Buffers**:\n    - 250 m: Mean = 1200.09, Min = 8.52, Max = 3089.27\n    - 500 m: Mean = 3433.24, Min = 27.98, Max = 8411.30\n    - 750 m: Mean = 6673.97, Min = 61.44, Max = 16160.06\n    - 1000 m: Mean = 10800.93, Min = 116.43, Max = 25562.41\n    - 1500 m: Mean = 21513.08, Min = 314.70, Max = 51698.95\n\nThese statistics provide a summary of the dataset, highlighting central tendencies, dispersions, and the range of values for each numeric attribute."
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         }
            ]

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

# if st.session_state.hypotheses:
#     st.subheader("Current Hypotheses")
#     for idx, hypo in enumerate(st.session_state.hypotheses, start=1):
#         st.markdown(f"**{idx}.** {hypo}")

if prompt := st.chat_input("Paste your hypotheses here",
                           accept_file=True):
    
    print(prompt)

    if prompt.text:
        st.sesssion_state.hypotheses.append(prompt.text)
    
    if prompt.files:
        file = prompt.files[0]
        file.seek(0)
        content = file.read().decode("utf-8")
        st.session_state.hypotheses.append(content)
        st.success("Hypotheses uploaded successfully!")

if st.button("Clear all hypotheses"):
    st.session_state.hypotheses.clear()
    st.success("All hypotheses cleared.")
    st.rerun()

# Optional: button to clear session state
if st.button("Refine hypotheses with the assistant"):

    print("Button clicked.")
    
    print(f"Hypotheses, type: {type(st.session_state.hypotheses)},\n{st.session_state.hypotheses}")
    print(f"Data summary: type: {type(st.session_state.data_summary)},\n{st.session_state.data_summary}")

    hypotheses_text = "\n\n".join(st.session_state.hypotheses)
    data_summary_text = "\n\n".join(item["content"] for item in st.session_state.data_summary)
    combined_text = f"{hypotheses_text}\n\n---\n\n{data_summary_text}"
    print(combined_text)
    st.markdown(combined_text)

    with st.spinner("Refining hypotheses ..."):
        print("Refining hypotheses ...")
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": combined_text
                }
            ],
        )

        st.write(stream)
    
        st.session_state.hypotheses.append(stream.choices[0].message.content)

    st.success("Have a look at the refined hypotheses.")

st.write(st.session_state.hypotheses)