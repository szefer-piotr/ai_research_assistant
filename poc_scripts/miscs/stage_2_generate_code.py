import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
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
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs
    )

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

## Set the page layout
st.set_page_config(page_title="Research assistant", 
                   page_icon=":robot:",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'Get Help': 'https://www.extremelycoolapp.com/help',
                       'Report a bug': "https://www.extremelycoolapp.com/bug",
                       'About': "# This is a header. This is an *extremely* cool app!"
                       })

st.markdown("### Research assistants :tada:")

## Set the session ststes
if "current_instruction" not in st.session_state:
    st.session_state.current_instruction = "Upload your data and your hypotheses"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
# if "data_summary" not in st.session_state:
#     st.session_state.data_summary = []
if "data_summary" not in st.session_state:
    st.session_state.data_summary = {}
if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = []
if "hypotheses_approved" not in st.session_state:
    st.session_state.hypotheses_approved = False
if "analysis_steps" not in st.session_state:
    st.session_state.analysis_steps = {}
if "files" not in st.session_state:
    st.session_state.files = {}
if "file_id" not in st.session_state:
    st.session_state.file_id = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = []
if "refine_hypotheses_button_pressed" not in st.session_state:
    st.session_state.refine_hypotheses_button_pressed = False


# data_summary_instructions = """
# Summarize the provided data. 
# Analyze each column name and infer its description based on its name.
# For each column provide the column name, its type (e.g. numeric, boolean, categorical) and basic statistics:
# - for numerical variables provide ranges, and basic statistics: mean, median, and quantiles.
# - for categorical or ordinal variables provide names of each unique values.
# Provide these for every column, not just exmples! Don't number each column. 
# Return one block of text.
# Provide this detailed analysis for EVERY SINGLE COLUMN in the dataset. 
# Every column name need to be in your summary! Do not just give examples and generalize!
# """

data_summary_instructions = """
Summarize the provided data.
Provide column names and infer what these names can mean.
For each column provide its name, description, and data type.
Return a dictionary with number of elements equal to the number of columns with a following schema:
{"column_name": <<name of the analysed column>> {"description": <<inferred description of the column>>,"type": <<type of the data>>, "unique_value_count": <<unique value count>>}},
"""

hypotheses_refinining_instructions = """
Based on the provided data summary perform a critical analysis of the provided hypotheses.
Analyze whether the hypotheses can be tested using the provided data. Voice any issues the user may have.
Suggest a refined version of each hypothesis. A hypothesis should be logical, should contain a metric that will be used to test it, and should define what sort of change can be expected in the metric.
You should search the web to inform the proposed and refined hypotheses.
"""

data_summary_assistant = client.beta.assistants.create(
                    name="Data Summarizing Assistant",
                    temperature=0,
                    instructions=data_summary_instructions,
                    tools=[{"type": "code_interpreter"}],
                    model="gpt-4o"
                )

code_execution_assistant = client.beta.assistants.create(
                    model="gpt-4o",
                    name="Code Execution Assistant",
                    instructions="Write code to execute the analysis step.",
                    tools=[{"type": "code_interpreter"}],
                    tool_resources={"code_interpreter": {
                        "file_ids": [file_id for file_id in st.session_state.file_id]}
                    }       
                )

# Utility functions
def chat_response(
    client,
    model: str,
    system_role_content,
    user_role_content,
    output_schema_dict: dict = None,
) -> list:
    '''
    Returns a response from OpenAI using the response API.

    Agrs:
        client: OpenAI client
        ...

    '''
    response = client.responses.create(
        model=model,
        inpout=[
            {"role": "system", "content": system_role_content},
            {"role": "system", "content": user_role_content}
        ],
        text=output_schema_dict
    )

    if output_schema_dict:
        return(json.loads(response.output_text))
    else:
        return(response.output_text)


# def assistant_response(
#     client,
#     thread_id,
#     assistant_id,
#     # system_role_content,
#     user_message: str = None,
#     # output_schema_dict: dict = None,
#     return_code = True,
# ) -> list:
#     '''
#     Returns a response from OpenAI using the Assistants API with a code interpreter.
#     Agrs:
#         client: OpenAI client
#         ...

#     '''
#     if user_message is not None:
#         message = client.beta.threads.messages.create(
#             thread_id=thread_id,
#             role="user",
#             content=user_message
#         )
    
#     client.beta.threads.update(
#         thread_id=st.session_state.thread_id,
#         tool_resources={"code_interpreter": {
#             "file_ids": [file_id for file_id in st.session_state.file_id]}}
#     )

#     stream = client.beta.threads.runs.create(
#         thread_id=thread_id,
#         assistant_id=assistant_id,
#         tool_choice={"type": "code_interpreter"},
#         stream=True
#     )

#     print(f"[INFO] Assistant response: {response}")

#     return response

    # stream = client.beta.threads.runs.create(
    #     thread_id=thread_id,
    #     assistant_id=assistant_id,
    #     tool_choice={"type": "code_interpreter"},
    #     stream=True
    # )

    # assistant_output = []               
    
    # # stream_handler(event)
    
    # for event in stream:
    #     stream_handler(assistant_output, event)

    #     if isinstance(event, ThreadMessageCreated):
    #         assistant_output.append({"type": "text",
    #                             "content": ""})
    #         assistant_text_box = st.empty()

    #     elif isinstance(event, ThreadMessageDelta):
    #         if isinstance(event.data.delta.content[0], TextDeltaBlock):
    #             assistant_text_box.empty()
    #             assistant_output[-1]["content"] += event.data.delta.content[0].text.value
    #             assistant_text_box.markdown(assistant_output[-1]["content"])

    # return(assistant_output)

def stream_handler(assistant_output):
    '''
    This function handles the stream.Resturns an assistant output, that is a list of dictionaries 
    that can be code, image or text.
    '''
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


# st.session_state.messages.append({"role": "assistant", "items": assistant_output})


st.write(
    st.session_state.current_instruction
)

prompt = st.chat_input("Ask me anything")

if st.session_state.data_summary and st.session_state.refine_hypotheses_button_pressed:
    
    with st.expander("Raw data and hypotheses"):
        st.markdown(f'<span style="font-size: 10px;">{st.session_state.combined_text}</span>', unsafe_allow_html=True)
    
    st.selectbox("Select a hypohthesis to work on.", options=["A","B","C"])
    
    # Display the message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input for user message
    if prompt := st.chat_input("Select a hypothesis that you would like to work on."):
        pass

    
        

elif st.session_state.data_summary:
    # print(st.session_state.data_summary)
    # with st.chat_message(st.session_state.data_summary["role"]):
    #     for item in st.session_state.data_summary["items"]:
    #         if item["type"] == "code_input":
    #             st.code(item["content"])
    #         elif item["type"] == "text":
    #             st.write(item["content"])
    #         elif item["type"] == "file":
    #             dataframe = pd.read_csv(item["file"])
    #             st.write(dataframe)
    button = st.button("Refine hypotheses with Assistant.")
    if button:
        with st.spinner("Refining hypotheses..."):
            refined_hypotheses = client.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": "Extract individual hypotheses from the text provided by the user and refine them one by one."},
                    {"role": "user", "content": st.session_state.combined_text}
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
            st.session_state.refined_hypotheses = refined_hypotheses
            st.session_state.refine_hypotheses_button_pressed = True
            st.session_state.messages.append({"role": "assistant", "content": "Lets work on your hypotheses!"})
            st.rerun()

# WHEN FILE IS LOADED
if st.session_state.file_uploaded:
    # Get the file names
    file_names = list(st.session_state.files.keys())[0]
    file_obj = st.session_state.files[file_names]
    file_obj.seek(0)
    
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Error reading the file: {e}")

    sidebar = st.sidebar
    sidebar.subheader("Data preview:")
    sidebar.write(f"File name: {file_names}")
    sidebar.dataframe(df)

    if not st.session_state.refine_hypotheses_button_pressed:

        button = st.button("Summarize the data with Assistant")

        if button:
            # Update the thread with the uploaded csv files.
            client.beta.threads.update(
                thread_id=st.session_state.thread_id,
                tool_resources={
                    "code_interpreter": {"file_ids" :
                        [
                            file_id for file_id in st.session_state.file_id
                        ]
                    }
                }
            )

            st.toast(f"Thread {st.session_state.thread_id} updated with the {file_names} file.")

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

                st.session_state.data_summary = {"role":"assistant", 
                                                "items": assistant_output}

            st.success("Data summarized successfully!")
            st.rerun()
            


if st.session_state.hypotheses:
    sidebar.title("Hypotheses")
    sidebar.markdown(st.session_state.hypotheses[0])

# WHEN FILE IS NOT LOADED YET
if not st.session_state.file_uploaded:
    # Build sidebar menu
    sidebar = st.sidebar
    sidebar.title("Files uploaded.")
    uploaded_file = sidebar.file_uploader("Upload a file", type=["csv"])

    if uploaded_file is not None:
        # Check if the file has already been uploaded
        if uploaded_file.name in st.session_state.files.keys():
            print(f"Uploaded file name: {uploaded_file.name}")
            st.warning("This file has already been uploaded.")
            # st.rerun()

        else: 
            # Display the uploaded file
            df = pd.read_csv(uploaded_file)
            sidebar.dataframe(df)

            st.toast("Dataset uploaded successfully")

            # Save the uploaded file in a dictionary
            st.session_state.files[uploaded_file.name] = uploaded_file
            # Change the boolean indicator
            st.session_state.file_uploaded = True
            
            # Create a thread for the user whith the uploaded file
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
            
            print(f"\nThread created with an ID {st.session_state.thread_id}")
            st.toast(f"Thread {thread.id} created.")

            # Create an openai file
            for file in st.session_state.files:
                openai_file = client.files.create(
                    file=st.session_state.files[file],
                    purpose="assistants"
                )
                st.session_state.file_id.append(openai_file.id)

            # Rerun the script to move to the next step
            # st.rerun()

if not st.session_state.hypotheses:
    sidebar.title("Hypotheses")
    uploaded_hypotheses = sidebar.file_uploader("Upload your hypotheses", type=["txt"])
    if uploaded_hypotheses is not None:
        content = uploaded_hypotheses.read().decode("utf-8")
        st.session_state.hypotheses.append(content)
        st.toast("Hypotheses uploaded successfully!")
        st.session_state.current_instruction = "Use dataset to refine the hypotheses."
        time.sleep(1)
        st.rerun()

if st.session_state.hypotheses and st.session_state.data_summary:
    
    for item in st.session_state.data_summary["items"]:
        if item["type"] == "text":
            data_summary_text = item["content"]

    hypotheses_text = "\n\n".join(st.session_state.hypotheses)
    # data_summary_text = st.session_state.data_summary
    combined_text = f"## The Data Summary \n\n{hypotheses_text}\n\n---\n\n{data_summary_text}"
    with st.expander("Hypotheses and data summary."):
        st.markdown(combined_text)
    st.session_state.combined_text = combined_text