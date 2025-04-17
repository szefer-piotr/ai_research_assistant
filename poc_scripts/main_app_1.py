import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
import base64
import openai

# from poc_scripts.assistant_event_handlers import EventHandler

from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseCreatedEvent,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    # AnnotationURLCitation
)

from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta,
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
When using the code interpreter ALWAYS load FULL dataset to have a complete view of the data.
Do not use `head()` when examining the data!
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
Task: 
- Refine the provided hypotheses further based on the daset description and refined hypotheses.
- The hypotheses should be more specific and actionable.
- Always search the web to review the latest scientific literature that can help refine the hypotheses.
- Search for the cited papers in web and provide links to the cited literature.
- In your response ALWAYS provide link to the cited papers.
- From your web search always provide a short summary of the information you found.
- Provide your thought process and refine the hypotheses step by step.
- The output should be a list of refined hypotheses.
"""

code_generation_instructions = """## Role
You are an exert in writing Python code for ecological data analysis.
## Task
Write and execute code when necessary.
Do not write code if not explicitly asked to!
When you write the code describe staps taken and provide interpretation of the results.
## Requirements
- Write clear code.
- Break the code into simple and easily executable steps.
"""

data_summary_assistant = client.beta.assistants.create(
    name="Data Summarizing Assistant",
    temperature=0,
    instructions=data_summary_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
)

code_execution_assistant = client.beta.assistants.create(
    name="Code Execution Assistant",
    temperature=1,
    instructions=code_generation_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
)

SIMPLE_SCHEMA = {"type": "object",
                 "properties": {
                     "hypothesis_title": {
                         "type": "string"
                         },
                    "refinement_justification": {
                        "type": "string"
                        },
                    "refined_hypothesis": {
                        "type": "string"
                        },
                    "references": {
                        "type": "array", 
                        "items": {
                            "type": "string"
                            }
                        },
                    },
                "required": ["hypothesis_title", "refinement_justification", "refined_hypothesis", "references"],
                "additionalProperties": False
                }

REFINED_HYPOTHESES_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": SIMPLE_SCHEMA
            }
        },
    "required": ["hypotheses"],
    "additionalProperties": False
    }

def main():

    # Force a font accross the app
    # st.markdown("""
    #     <link href="https://fonts.googleapis.com/css2?family=Open+Sans&display=swap" rel="stylesheet">
    #     <style>
    #     * {
    #         font-family: 'Open Sans', sans-serif !important;
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)

        # Header with soulless (monospace) font
    st.markdown("""
        <h1 style='text-align: center; font-family: monospace; letter-spacing: 2px;'>
        RESEARCH ASSISTANT
        </h1>
        """, unsafe_allow_html=True)

    # Initialize session state for files
    if "csv_file" not in st.session_state:
        st.session_state["csv_file"] = {}
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = {}
    if "file_ids" not in st.session_state:
        st.session_state["file_ids"] = []
    if "hypotheses" not in st.session_state:
        st.session_state["hypotheses"] = ""
    if "refined_hypotheses" not in st.session_state:
        st.session_state["refined_hypotheses"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    # This boolean flag controls whether we hide the file overview
    if "processing" not in st.session_state:
        st.session_state["processing"] = False
    if "refinement" not in st.session_state:
        st.session_state["refinement"] = False
    if "planning" not in st.session_state:
        st.session_state["planning"] = False
    if "web_search_calls" not in st.session_state:
        st.session_state["web_search_calls"] = []
    if "data_processing_output" not in st.session_state:
        st.session_state["data_processing_output"] = []
    if "analysis_plan" not in st.session_state:
        st.session_state["analysis_plan"] = {}
    if "code_execution" not in st.session_state:
        st.session_state["code_execution"] = {}

    # Only show the chat messages if we are already processing
    if st.session_state["processing"]:
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





###################################################################################################


        button = st.sidebar.button("Refine hypotheses with LLM.")

        # pprint.pprint(st.session_state.messages)

        data = st.session_state.messages[0]
        text_contents = [item["content"] for item in data["items"] if item["type"] == "text"]
        joined_text = "\n".join(text_contents)

        if button:
            collected_citations = []
            with st.chat_message("assistant"):

                # response_no_stream = client.responses.create(
                #     model="gpt-4o",
                #     tools=[{"type": "web_search_preview"}],
                #     input=joined_text,
                #     instructions=refine_hypotheses_instructions,
                # )

                # print(f"\n\n[RESPONSE NO STREAM] {response_no_stream.output_text}\n\n")

                # st.session_state.messages.append({"role": "assistant", "items": [response]})

                response = client.responses.create(
                    # model="gpt-4o",
                    model="gpt-4o",
                    input=joined_text,
                    instructions=refine_hypotheses_instructions,
                    tools=[{"type": "web_search_preview"}],
                    tool_choice='required',
                    stream=True,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "refined_hypotheses",
                            "schema": REFINED_HYPOTHESES_SCHEMA,
                            "strict": True
                        }
                    },
                    temperature=0.1
                )

                assistant_output = []

                print("NEW RESPONSE:")

                for event in response:                    
                    print(f"{event}")
                    
                    if isinstance(event, ResponseCreatedEvent):
                        assistant_output.append({"type": "text", "content": ""})
                        assistant_text_box = st.empty()
                    
                    elif isinstance(event, ResponseTextDeltaEvent):
                        if event.type == 'response.output_text.delta':
                            assistant_text_box.empty()
                            assistant_output[-1]["content"] += event.delta
                            assistant_text_box.markdown(assistant_output[-1]["content"])

                    elif isinstance(event, ResponseCompletedEvent):
                        completed_response = event.response
                        for item in completed_response.output:
                            if isinstance(item, ResponseOutputMessage):
                                for content_block in item.content:
                                    if isinstance(content_block, ResponseOutputText):
                                        print(f"\nRESPONSE OUTPUT TEXT: {content_block}\n")
                                        for ann in content_block.annotations:
                                            collected_citations.append((ann.title, ann.url))

                            # # Get the web search result
                            # web_search_result = client.web_searches.retrieve(web_search_call.id)
                            # print(web_search_result)

                            # # Create a new item for the assistant output
                            # assistant_output.append({
                            #     "type": "text",
                            #     "content": f"Web search result: {web_search_result.result}"
                            # })

                st.session_state.messages.append({"role": "assistant", "items": assistant_output})
                st.session_state.refined_hypotheses.append(assistant_output)
                if collected_citations:
                    st.session_state["web_search_calls"].append(collected_citations)
                
                
                # Change the app state
                st.session_state["refinement"] = True
                st.session_state["processing"] = False
        
            st.rerun()
    
    if st.session_state["refinement"]:   
        
        with st.sidebar.expander("Web Search Sources", expanded=True):
            if "web_search_calls" in st.session_state and st.session_state["web_search_calls"]:
                for i, citation in enumerate(st.session_state["web_search_calls"][0], start=1):
                    st.write(f"{i}. Title {citation[0]}")
                    st.write(f"Link: {citation[1]}")
                    # st.write(f"{i}. Title: {ws_call[0]}")
                    # st.write(ws_call[1])
                    # st.write(ws_call.url)
                    # Dump the entire call object, or just relevant parts
                    # st.json({
                    #     "id": ws_call.id,
                    #     "status": ws_call.status,
                    #     "type": ws_call.type,
                    #     # Add any extra fields if available
                    # })
            else:
                st.write("No web search calls made yet.")

        button = st.button("Prepare Analysis Plan.")
        
        if button:
            st.session_state["planning"] = True
            st.session_state["refinement"] = False
            st.rerun()

        
        print(f"\n[MESSAGES]: {st.session_state.messages}\n")
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
    
    if st.session_state["planning"]:


        st.header("Prepare Analysis Plan")
        # print(st.session_state.analysis_plan)

        refined_hypotheses = json.loads(st.session_state.refined_hypotheses[0][0]["content"])
        
        # print(refined_hypotheses)
        # print(type(refined_hypotheses))

        if len(st.session_state.analysis_plan) < len(refined_hypotheses['hypotheses']):
            print("Some hypotheses have the plan!")
            selected_hypothesis = st.selectbox("Select hypothesis", options=st.session_state.analysis_plan)
            if selected_hypothesis:
                hypothesis_to_display = st.session_state.analysis_plan[selected_hypothesis]
                st.header(selected_hypothesis)
                st.markdown(hypothesis_to_display)
                
                print(f"ANALYSIS PLAN LENGTH: {len(st.session_state.analysis_plan)}")
                print(f"REFINED HYPOTHESE LENGTH: {len(refined_hypotheses['hypotheses'])}")

        elif len(st.session_state.analysis_plan) == len(refined_hypotheses['hypotheses']):
            
            sidebar = st.sidebar

            selected_hypothesis = sidebar.selectbox(
                "Select hypothesis", 
                options=st.session_state.analysis_plan
            )

            button = sidebar.button("Run the analysis.", key=f"analysis_{selected_hypothesis}")

            if selected_hypothesis:
                hypothesis_to_display = st.session_state.analysis_plan[selected_hypothesis]
                # sidebar.header(selected_hypothesis)
                # sidebar.markdown(hypothesis_to_display)
                
                if button:
                    
                    stream = client.beta.threads.runs.create(
                        thread_id=st.session_state.thread_id,
                        assistant_id=code_execution_assistant.id,
                        instructions=f"""Execute the following analysis plan: {hypothesis_to_display}""",
                        tools=[{"type": "code_interpreter"}],
                        temperature=0,
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
                    st.session_state.code_execution[selected_hypothesis] = assistant_output 

                    print("***"*10)
                    for message in openai.beta.threads.messages.list(st.session_state.thread_id):
                        print(f"\n\n{message}")

        else:
            st.info("Create plan for an analysis.")
        #     

        with st.sidebar:
            for hypothesis in refined_hypotheses["hypotheses"]:
                st.header(hypothesis["hypothesis_title"])
                st.markdown(hypothesis["refined_hypothesis"])
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
                button = st.button("Prepare Analysis Plan", key=hypothesis["hypothesis_title"])

                plan_generation_instructions = f"""
                # Data summary
                {st.session_state["data_processing_output"]}
                # Task
                Prepare an analysis plan for the following hypothesis:
                {hypothesis["refined_hypothesis"]}.
                # Requirements
                - For each step of a plan, **consider possible improvements** in statistical methods.
                - Anticipate **common statistical issues**, such as:
                - Small sample sizes
                - Multicollinearity
                - Spatial autocorrelation
                - Provide **detailed solutions** for handling these issues.
                - The plan should be **clear, unambiguous, and contain all necessary steps**.
                - Instead of vague instructions like *"deal with missing values appropriately"*, provide **specific** actions.
                - Decline questions **outside your scope** and remind students of the **topics you cover**.
                - **Critically analyze** each step and **ALWAYS** provide methods best suited to the data.
                - Use exact dataset names and **necessary column names** from the **data summary**.
                - The response should be in the form of a **numbered plan**, consisting of **simple, executable steps**.
                - **Break down** complex steps into **smaller, clear and atomic sub-steps**.
                """

                # print(f"\nPLAN: \n {plan_generation_instructions}")

                if button:
                    response = client.responses.create(
                        model="gpt-4o",
                        input=plan_generation_instructions,
                        stream=False,
                        temperature=0.1
                    )

                    st.session_state.analysis_plan[hypothesis["hypothesis_title"]] = response.output_text

                    print(f"\nTHE ANALYSIS PLAN: {st.session_state.analysis_plan}\n")

                    st.rerun()



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
            print(thread)
            st.session_state.thread_id = thread.id

            with st.chat_message("assistant"):
                
                stream = client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=data_summary_assistant.id,
                    tool_choice={"type": "code_interpreter"},
                    temperature=0,
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
                st.session_state.data_processing_output.append(assistant_output)

            # Rerun so that on the next load, "processing" is True and we show the streamed messages only
            st.rerun()


if __name__ == "__main__":
    main()
