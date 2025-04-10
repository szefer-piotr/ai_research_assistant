import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json

import pandas as pd

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
    # st.session_state.data_summary = [
    #     {
    #         "type":"text",
    #         "content":"The dataset consists of 61 columns. Here are the column names along with their data types:\n\n1. **INDEX_OF_INDIVIDUALS**: int\n2. **Bee.species**: object (string)\n3. **Species.code**: object (string)\n4. **Sex**: object (string)\n5. **Site.number**: object (string)\n6. **Year**: int\n7. **Month**: object (string)\n8. **Day**: int\n9. **Family**: object (string)\n10. **Social.behavior**: object (string)\n11. Additional columns related to impervious surface area and population density, all appearing as strings at first sight but potentially representing numerical values. These columns include:\n    - **Impervious.surface.area.in.buffer.250.m.[mean]**\n    - **Impervious.surface.area.in.buffer.500.m.[mean]**\n    - **Impervious.surface.area.in.buffer.750.m.[mean]**\n    - **Impervious.surface.area.in.buffer.1000.m.[mean]**\n    - **Impervious.surface.area.in.buffer.1500.m.[mean]**\n    - **Population.density.in.buffer.250.m**\n    - **Population.density.in.buffer.500.m**\n    - **Population.density.in.buffer.750.m**\n    - **Population.density.in.buffer.1000.m**\n    - **Population.density.in.buffer.1500.m**\n\nLet's convert any necessary columns to numeric data types and then provide basic statistics for the numerical columns."},
    #     {
    #         "type":"text",
    #         "content":"Here are the basic statistics for the numeric columns in the dataset:\n\n1. **INDEX_OF_INDIVIDUALS**: \n   - Mean: 2882\n   - Min: 1\n   - Max: 5763\n\n2. **Year**:\n   - Mean: 2018.62\n   - Min: 2018\n   - Max: 2019\n\n3. **Day**:\n   - Mean: 10.15\n   - Min: 1\n   - Max: 28\n\n4. **Lifespan (month)**:\n   - Mean: 4.79\n   - Min: 1\n   - Max: 8\n\n5. **Shortest Distance Between Sites (m)**:\n   - Mean: 4684.31\n   - Min: 527.72\n   - Max: 20068.52\n\n6. **Year of Research**:\n   - Mean: 2018.62\n   - Min: 2018\n   - Max: 2019\n\n7. **Coverage of Bee Food Plant Species (%)**:\n   - Mean: 41.77\n   - Min: 17\n   - Max: 65\n\n8. **Floral Richness**: \n   - Mean: 71.27\n   - Min: 32\n   - Max: 130\n\n9. **Alien Floral Richness (%)**:\n   - Mean: 30.19\n   - Min: 21.95\n   - Max: 39.71\n\n10. **Native Floral Richness (%)**:\n    - Mean: 54.03\n    - Min: 32.22\n    - Max: 75.76\n\n11. **Impervious Surface Area in Buffers (mean values)**:\n    - 250 m: Mean = 31.60, Min = 0.51, Max = 81.18\n    - 500 m: Mean = 29.33, Min = 0.20, Max = 73.53\n    - 750 m: Mean = 28.68, Min = 0.12, Max = 67.17\n    - 1000 m: Mean = 27.94, Min = 0.26, Max = 64.89\n    - 1500 m: Mean = 26.32, Min = 0.59, Max = 61.13\n\n12. **Population Density in Buffers**:\n    - 250 m: Mean = 1200.09, Min = 8.52, Max = 3089.27\n    - 500 m: Mean = 3433.24, Min = 27.98, Max = 8411.30\n    - 750 m: Mean = 6673.97, Min = 61.44, Max = 16160.06\n    - 1000 m: Mean = 10800.93, Min = 116.43, Max = 25562.41\n    - 1500 m: Mean = 21513.08, Min = 314.70, Max = 51698.95\n\nThese statistics provide a summary of the dataset, highlighting central tendencies, dispersions, and the range of values for each numeric attribute."
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      }
    #         ]

if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = []

if "analysis_steps" not in st.session_state:
    st.session_state.analysis_steps = {}

# Store the file IDs for the thread
if "file_id" not in st.session_state:
    st.session_state.file_id = []

# Store the thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = []

if "hypotheses_approved" not in st.session_state:
    st.session_state.hypotheses_approved = False

if not st.session_state.hypotheses_approved:
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

# If the hypotheses are approved, display them
else:
    # with st.sidebar:
    st.subheader("Select a Hypothesis")
    selected_hypothesis = st.selectbox(
        "Select a hypothesis to work on", options=[
            st.session_state.hypotheses["hypotheses"][i]["title"] for i in range(len(st.session_state.hypotheses["hypotheses"]))
        ]
    )

    hypotheses_text = {}
    
    for hypothesis in st.session_state.hypotheses["hypotheses"]:
        title = hypothesis['title']
        steps = [step['step'] for step in hypothesis['steps']]
        hypotheses_text[title] = steps
    
    if selected_hypothesis:
        #  st.write(selected_hypothesis)
        st.write(hypotheses_text[selected_hypothesis])
        
        if st.button("Create an analysis plan"):
            with st.spinner("Creating an analysis plan ..."):
                analysis_plan = client.responses.create(
                    model="gpt-4o-2024-08-06",
                    input=[
                        {"role": "system", "content": "Generate an analysis plan for the hypothesis."},
                        {"role": "user", 
                        "content": f"Here is the data summary: {st.session_state.data_summary}. For each part of the hypothesis: {hypotheses_text[selected_hypothesis]}, write individual analysis step details for later implementation in code. Steps have to simle and programmatically executable"}
                    ],
                    text={
                        "format": {
                        "type": "json_schema",
                        "name": "analysis_plan",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "analysis_plan": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": { "type": "string" },
                                            "steps": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "step": { "type": "string" }
                                                    },
                                                    "required": ["step"],
                                                    "additionalProperties": False
                                                }
                                            }
                                        },
                                        "required": ["title", "steps"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["analysis_plan"],
                            "additionalProperties": False
                        },
                        "strict": True
                        }
                    }
                )
            st.success("Analysis plan created successfully!")
            st.session_state.analysis_steps[selected_hypothesis] = (json.loads(analysis_plan.output_text))
            # analysis_plan_output = json.loads(analysis_plan.output_text)
            # st.write(analysis_plan_output)

    sideb = st.sidebar

    sideb.markdown("## Analysis Steps")
    hypothesis_to_generate_code = sideb.selectbox("Select a hypothesis to work on", options=list(st.session_state.analysis_steps.keys()))

    # with st.sidebar:
    #     st.markdown("### Analysis Steps")
    #     # st.write(st.session_state.analysis_steps)
    #     # hypothesis_to_generate_code = st.selectbox("Select a hypothesis to work on", options=list(st.session_state.analysis_steps.keys()))
        
    if hypothesis_to_generate_code:
        st.markdown(f"### {hypothesis_to_generate_code}")
        st.write(st.session_state.analysis_steps[hypothesis_to_generate_code])
        print(f"Analysis steps: {json.dumps(st.session_state.analysis_steps[hypothesis_to_generate_code])}")
        
        if st.button("Run the step."):
            with st.spinner("Running the step ..."):
                
                # Create a thread and attach the hypothesis steps and data summary
                code_execution_assistant = client.beta.assistants.create(
                    model="gpt-4o",
                    name="Code Execution Assistant",
                    instructions="Write code to execute the analysis step.",
                    tools=[{"type": "code_interpreter"}],
                    tool_resources={"code_interpreter": {
                        "file_ids": [file_id for file_id in st.session_state.file_id]}
                    }       
                )
                
                step_text = json.dumps(st.session_state.analysis_steps[hypothesis_to_generate_code])

                # Create a message on the thread
                message = client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content=step_text
                )

                stream = client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=code_execution_assistant.id,
                    tool_choice={"type": "code_interpreter"},
                    stream=True
                )

                assistant_output = []

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

            st.success("Code generated successfully!")


    st.stop()

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
    # data_summary_text = "\n\n".join(item["content"] for item in st.session_state.data_summary)
    # data_summary_text = "\n\n".join(st.session_state.data_summary)
    data_summary_text = st.session_state.data_summary
    combined_text = f"{hypotheses_text}\n\n---\n\n{data_summary_text}"
    print(combined_text)
    st.markdown(combined_text)

    with st.spinner("Refining hypotheses ..."):
        print("Refining hypotheses ...")

        response = client.responses.create(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "Extract individual hypotheses and refine them one by one."},
                {"role": "user", "content": combined_text}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "hypotheses",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "hypotheses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": { "type": "string" },
                                        "steps": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "step": { "type": "string" }
                                                },
                                                "required": ["step"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["title", "steps"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["hypotheses"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )

        print(json.loads(response.output_text))
        st.markdown(json.loads(response.output_text))
        st.session_state.hypotheses.append(json.loads(response.output_text))

    st.success("Have a look at the refined hypotheses.")
    
if st.button("Save refined hypotheses"):
    # Saves only the last hypothesis
    st.session_state.hypotheses = st.session_state.hypotheses[-1]
    st.session_state.hypotheses_approved = True
    st.success("Hypotheses saved successfully!")

with st.sidebar:
    if st.button("Prepare the analysis plan"):
        st.session_state.messages.append({"role": "assistant",
                                        "items": [
                                            {"type": "text",
                                            "content": "Let's move on to the next step: preparing the analysis plan."}
                                        ]})

    st.rerun()