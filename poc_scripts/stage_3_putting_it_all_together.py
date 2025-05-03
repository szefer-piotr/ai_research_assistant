import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import pandas as pd

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

## Set the page layout
st.set_page_config(page_title="Research assistant", 
                   page_icon=":tada:",
                   layout="wide")


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

processing_files_instruction = """
Task: Based on the provided data summary, perform a critical analysis of each given hypothesis.
Assess Testability: Determine whether each hypothesis can be tested using the provided data. Justify your reasoning by referencing the relevant variables and their formats.
Identify Issues: Highlight any conceptual, statistical, or practical issues that may hinder testing — e.g., vague metrics, missing data, confounding variables, or unclear expected effects.
Refine Hypotheses: Suggest a clear and testable version of each hypothesis. Each refined hypothesis should:
Be logically structured and grounded in the data.
Include a specific metric to be analyzed.
Indicate the direction or nature of the expected change (e.g., increase/decrease, positive/negative correlation).
Be framed in a way that is statistically testable.
Support with External Knowledge: If needed, search the web or draw from scientific literature to refine or inform the hypotheses.
"""

refinig_instructions = """
You are an expert in ecological research and hypothesis development. Your task is to help refine the hypotheses provided by the user.
Instructions:
Critically analyze the dataset shared by the user.
Evaluate each hypothesis to determine whether it:
Aligns with ecological theory or known patterns.
Can be tested using the available data (based on variable types, structure, and coverage).
If necessary, search the web for up-to-date ecological research or contextual knowledge to inform the refinement process.
For each hypothesis, suggest a refined version that:
Clearly defines the expected relationship or effect.
Includes specific variables or metrics from the dataset.
Is phrased in a way that is statistically testable.
Important Constraints:
Do not respond to any questions unrelated to the provided hypotheses.
Use domain knowledge and data-driven reasoning to ensure each refined hypothesis is grounded in ecological theory and evidence.
Output Format (for each hypothesis):
Original Hypothesis:
Can it be tested? (Yes/No with explanation)
Issues or concerns:
Refined Hypothesis:
Supporting context (optional, if external sources were used):
"""

analyses_step_generation_instructions = """
## Role
- You are an **expert in ecological research and statistical analysis**, with proficiency in **Python**.
- You must apply the **highest quality statistical methods and approaches** in data analysis.
- Your suggestions should be based on **best practices in ecological data analysis**.
- Your role is to **provide guidance, suggestions, and recommendations** within your area of expertise.
- Students will seek your assistance with **data analysis, interpretation, and statistical methods**.
- Since students have **limited statistical knowledge**, your responses should be **simple and precise**.
- Students also have **limited programming experience**, so provide **clear and detailed instructions**.
"""

step_execution_assistant_instructions = """
## Role
You are an expert in ecological research and statistical analysis in Python. 
## Task
- execute the analysis plan provided by the user STEP BY STEP. 
- Write code in Python for each step to of the analysis plan from the beginning to the end.
- execute code, write description and short summary forr all of the steps.
"""

step_execution_instructions = """
Execute in code every step of the analysis plan.
"""


data_summary_assistant = client.beta.assistants.create(
    name="Data Summarizing Assistant",
    temperature=0,
    instructions=data_summary_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
    )


analysis_assistant = client.beta.assistants.create(
    name="Analysis Assistant",
    temperature=0,
    instructions=step_execution_assistant_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
    )


if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False
if "hypotheses_uploaded" not in st.session_state:
    st.session_state.hypotheses_uploaded = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = []
if "analysis_execution_thread_id" not in st.session_state:
    st.session_state.analysis_execution_thread_id = []
if "files" not in st.session_state:
    st.session_state.files = {}
if "file_ids" not in st.session_state:
    st.session_state.file_ids = []
if "data_summary" not in st.session_state:
    st.session_state.data_summary = ""
if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = ""
# Comment this after checking the plan.
if "hypotheses_refined" not in st.session_state:
    st.session_state.hypotheses_refined = False
if "refined_hypotheses" not in st.session_state:
    st.session_state.refined_hypotheses = {}
if "approved_analyses_plans" not in st.session_state:
    st.session_state.approved_analyses_plans = []
if "updated_hypotheses" not in st.session_state:
    st.session_state.updated_hypotheses = {}
if "app_state" not in st.session_state:
    st.session_state.app_state = "upload"
if "approved_hypotheses" not in st.session_state:
    st.session_state.approved_hypotheses = []
if 'analysis_outputs' not in st.session_state:
    st.session_state['analysis_outputs'] = {}





# Header with soulless (monospace) font
st.markdown("""
<h1 style='text-align: center; font-family: monospace; letter-spacing: 2px;'>
RESEARCH ASSISTANT
</h1>
""", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns(2)






########################################################################################
############ STAGE 1: TXT upload on the right, # CSV upload on the left ################
########################################################################################

if st.session_state.app_state == "upload":

    with col1:
        st.markdown(
            """
            <span style='font-size:16px; font-weight:600;'>📄 Upload your dataset</span>
            """, unsafe_allow_html=True)
        csv_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
        if csv_file:
            st.toast("CSV file uploaded!")
            # Optional preview
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            # Add the file to a dictionatry to refer to it by its name
            st.session_state.files[csv_file.name] = csv_file
            st.session_state.data_uploaded = True
            with st.expander("Data preview."):
                st.dataframe(df.head())


    with col2:
        st.markdown(
            """
            <span style='font-size:16px; font-weight:600;'>📝 Upload your hypotheses</span>
            """, unsafe_allow_html=True)
        txt_file = st.file_uploader("Choose a TXT file", type="txt", key="txt_uploader")
        if txt_file:
            st.toast("TXT file uploaded!")
            text = txt_file.read().decode("utf-8")
            st.session_state.hypotheses = text
            st.session_state.hypotheses_uploaded = True
            with st.expander("Hypotheses preview."):
                st.text_area("File content", text, height=182)

    if st.session_state.data_uploaded and st.session_state.hypotheses_uploaded:
        st.session_state.app_state = "processing"
        st.rerun()

# Three columns to center the button in the middle one
if st.session_state.app_state == "processing":

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🚀 Process Files"):        
            # Create a thread
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id

            # Create an openai file
            for file in st.session_state.files:
                openai_file = client.files.create(
                    file=st.session_state.files[file],
                    purpose="assistants"
                )
                st.session_state.file_ids.append(openai_file.id)
            
            # Update the thread with uploaded files
            client.beta.threads.update(
                    thread_id=st.session_state.thread_id,
                    tool_resources={
                        "code_interpreter": {"file_ids" :
                            [
                                file_id for file_id in st.session_state.file_ids
                            ]
                        }
                    }
                )
            
            with st.spinner("Refining your hypotheses..."):
                
                # Step I summarize the data and save the summary
                run = client.beta.threads.runs.create_and_poll(
                    thread_id=st.session_state.thread_id,
                    assistant_id=data_summary_assistant.id,
                    instructions=data_summary_instructions,
                    temperature=0
                )
                if run.status == 'completed':
                    messages = client.beta.threads.messages.list(
                        thread_id=st.session_state.thread_id
                    )

                    messages_list = list(messages)
                    
                    assistant_response = []
                    
                    for msg in messages_list:
                        for block in msg.content:
                            if block.type == 'text':
                                assistant_response.append(block.text.value)

                    st.session_state.data_summary = " ".join(assistant_response)
                    with st.expander("Data summary"):
                        st.write(st.session_state.data_summary)

                prompt = f"""
                Data summary preview: {st.session_state.data_summary}.\n\nHypotheses: {st.session_state.hypotheses}.\n\n {processing_files_instruction}
                Extract individual hypotheses from the text provided by the user and refine them one by one.            
                """

                # Step II refine hypotheses
                response = client.responses.create(
                    model="gpt-4o",
                    input=[{"role": "user",
                            "content": prompt}],
                    tools=[{"type": "web_search_preview"}],
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
                
                updated_hypotheses = json.loads(response.output_text)
                # Make a list of titles as keys
                hypotheses_title_keys_list = [hypothesis['title'] for hypothesis in updated_hypotheses['hypotheses']]            
                # Save the updated hypotheses to the session state
                st.session_state.updated_hypotheses = updated_hypotheses

                #############################################
                ## DEFINE FIELDS FOR THE HYPOTHESES OBJECT ##
                #############################################

                # Add empty fileds for chat and for the final hypothesis
                for i, hypothesis in enumerate(updated_hypotheses['hypotheses']):
                    st.session_state.updated_hypotheses["hypotheses"][i]['chat_history'] = []
                    st.session_state.updated_hypotheses["hypotheses"][i]['final_hypothesis'] = ""
                    st.session_state.updated_hypotheses["hypotheses"][i]['analysis_plan'] = []
                    st.session_state.updated_hypotheses["hypotheses"][i]['analysis_plan_chat_history'] = []
                    st.session_state.updated_hypotheses["hypotheses"][i]['analysis_plan_accepted'] = False
                    st.session_state.updated_hypotheses["hypotheses"][i]['plan_execution_chat_history'] = []
                    st.session_state.updated_hypotheses["hypotheses"][i]['plan_execution'] = []
                    
                    # Create an initial message for in the chat history
                    st.session_state.updated_hypotheses['hypotheses'][i]['chat_history'].append(
                        {
                            "role":"assistant", 
                            "content": f""""
                            Here is the refined hypothesis: {hypothesis['title']}. 
                            Here is the rationale for its refinement: {hypothesis['steps']}
                            """
                        }
                    )

                    st.write(st.session_state.updated_hypotheses["hypotheses"][i])

                # Success
                st.success("Done")
    with col2:            
        if st.button("Move to the next step"):
            st.session_state.app_state = "hypotheses_manager"        
            st.rerun()



###############################################
####### STAGE 2 - HYPOTHESIS MANAGER ##########
###############################################



if st.session_state.app_state == "hypotheses_manager":
    
    # print(f"Current app state {st.session_state.app_state}")
    # print(f"The hypotheses object: {st.session_state.updated_hypotheses['hypotheses']}")
    
    st.title("Hypothesis Manager")

    for i, hypothesis_obj in enumerate(st.session_state.updated_hypotheses["hypotheses"]):
        
        with st.expander(f"{hypothesis_obj['title']}"):

            if st.session_state.updated_hypotheses["hypotheses"][i]['final_hypothesis'] != "":
                refined_hypothesis = st.session_state.updated_hypotheses['hypotheses'][i]['final_hypothesis']
                st.markdown(refined_hypothesis)

            # Final hypothesis is empty
            elif st.session_state.updated_hypotheses["hypotheses"][i]['final_hypothesis'] == "":
                
                # Display the refined hypothesis text in steps
                for step_j, step_obj in enumerate(hypothesis_obj["steps"]):
                    st.markdown(f"**Step {step_j+1}:** {step_obj['step']}")

                # Display history of the previous conversations about this particular hypothesis.
                for msg in st.session_state.updated_hypotheses['hypotheses'][i]['chat_history']:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                prompt = st.chat_input(
                    "Discuss with the assistant to refine this hypothesis further.",
                    key=f"chat_input{i}")
                
                # Provide all the conversation history during the chat
                history = st.session_state.updated_hypotheses['hypotheses'][i]['chat_history']

                if prompt:
                    
                    with st.spinner("Thinking..."):
                        # Append the user message to the chat history
                        st.session_state.updated_hypotheses['hypotheses'][i]['chat_history'].append(
                            {"role":"user", "content": prompt}
                        )
                        
                        print(f"History in HYPOTHESIS MANAGER: {history}")
                        
                        response = client.responses.create(
                            model="gpt-4o",
                            instructions=refinig_instructions,
                            input=history,
                            tools=[{"type": "web_search_preview"}],
                            store=False
                            # [TODO] Add the schema here
                        )

                        # Append the LLM's response to the chat history
                        st.session_state.updated_hypotheses['hypotheses'][i]['chat_history'].append(
                        {"role":"assistant", "content": response.output_text}
                        )

                    st.rerun()
                
                button = st.button("Accept the refined hypotheses", key=f"button{i}")

                if button:
                    # Saves the last message from the history.
                    last_message = st.session_state.updated_hypotheses['hypotheses'][i]['chat_history'][-1]['content']

                    # Here the last hypotheses should be saved as accepted_hypotheses
                    st.session_state.updated_hypotheses["hypotheses"][i]['final_hypothesis'] = last_message
                    st.rerun()

                # pass
    if all(hypothesis['final_hypothesis'] != "" for hypothesis in st.session_state.updated_hypotheses["hypotheses"]):
        st.session_state.app_state = "plan_manager"
        st.rerun()








##################################################
######## STAGE 3: ANALYSIS PLAN MANAGER ##########
##################################################

if st.session_state.app_state == "plan_manager":
    
    final_hypotheses_list = [
        hypo['final_hypothesis'] for hypo in st.session_state.updated_hypotheses['hypotheses']]

    st.subheader("Analysis Plan Manager")

    # Loop over each hypothesis
    for i, hypothesis in enumerate(final_hypotheses_list):
        
        plan_steps_generation_input = f"""
                    Here is the data summary: {st.session_state.data_summary}\n
                    Here is the hypothesis: {hypothesis}.
                    """

        with st.expander(f"Accepted hypothesis {i+1}"):

            # Display the refined hypothesis text
            st.markdown(hypothesis)

            if st.session_state.updated_hypotheses['hypotheses'][i]['analysis_plan_chat_history'] == []:

                #### FIRST GENERAE A PLAN FOR THE ANALYSIS###############################################
                generate_button = st.button(f"Generate a plan to test the Hypothesis {i+1}.", key=f"generate_plan_{i}")
                    
                if generate_button:
                    response = client.responses.create(
                        model="gpt-4o",
                        temperature=0,
                        instructions=analyses_step_generation_instructions,
                        input=plan_steps_generation_input,
                        stream=False,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "analyses",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "analyses": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "title": {"type": "string"},
                                                    "steps": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "step": {"type": "string"}
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
                                    "required": ["analyses"],
                                    "additionalProperties": False
                                },
                                "strict": True
                            }
                        },
                        store=False
                    )
                    
                    # Save the newly generated plan into the message_history
                    new_plan = {"role": "assistant", "content": response.output_text}
                    st.session_state.updated_hypotheses['hypotheses'][i]['analysis_plan_chat_history'].append(new_plan)

                    # Rerun the script so that next time we detect the plan is generated
                    st.rerun()

            ########################################################################################
            
            # If the plan has not yet been accepted, show the message history
            
            if st.session_state.updated_hypotheses['hypotheses'][i]["analysis_plan_accepted"] != True:
                
                analysis_plan_chat_history = st.session_state.updated_hypotheses['hypotheses'][i]['analysis_plan_chat_history']
                
                if analysis_plan_chat_history != []:
                    # Display the message history for that hypothesis.
                    for message in analysis_plan_chat_history:
                        with st.chat_message(message['role']):
                            st.markdown(message['content'])

                    prompt = st.chat_input(
                        "Discuss with the assistant to refine this analysis plan.",
                        key=f"chat_input_plan_{i}")

                    if prompt:
                        
                        st.session_state.updated_hypotheses['hypotheses'][i]['analysis_plan_chat_history'].append(
                            {"role": "user", "content": prompt}
                            )
                        
                        history =  st.session_state.updated_hypotheses['hypotheses'][i]['analysis_plan_chat_history']

                        print(f"\n\n HISTORY:{history}")

                        response = client.responses.create(
                            model="gpt-4o",
                            temperature=0,
                            instructions=analyses_step_generation_instructions,
                            input=history,
                            stream=False,
                            text={
                                "format": {
                                    "type": "json_schema",
                                    "name": "analyses",
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "analyses": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "title": {"type": "string"},
                                                        "steps": {
                                                            "type": "array",
                                                            "items": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "step": {"type": "string"}
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
                                        "required": ["analyses"],
                                        "additionalProperties": False
                                    },
                                    "strict": True
                                }
                            },
                            store=False
                        )
                        
                        # Save the newly generated plan into the message_history
                        st.session_state.updated_hypotheses['hypotheses'][i]['analysis_plan_chat_history'].append(
                            {"role": "assistant", "content": response.output_text}
                            )
                        
                        st.rerun()

                     # Show the Accept button if a plan is generated
                    accept_button = st.button("Accept the plan", key=f"accept_plan_{i}")

                    if accept_button:
                        # On accept, store the last plan as final_hypothesis
                        st.session_state.updated_hypotheses['hypotheses'][i]["analysis_plan"] = analysis_plan_chat_history[-1]['content']
                        st.session_state.updated_hypotheses['hypotheses'][i]["analysis_plan_accepted"] = True
                        st.rerun()

            
            
            # Plan Accepted
            if st.session_state.updated_hypotheses['hypotheses'][i]["analysis_plan_accepted"] == True:
                # Display the analysis plan steps
                st.write(st.session_state.updated_hypotheses['hypotheses'][i]["analysis_plan"])
                

    if all(
        hypothesis["analysis_plan"] != "" 
        for hypothesis in st.session_state.updated_hypotheses["hypotheses"]
        ):

        if st.button("Move to the plan execution stage"):
            st.session_state.app_state = "plan_execution"
            st.rerun()








#######################################################
####### PLAN EXECUTION - ANALYSIS MANAGER #############
#######################################################

os.makedirs("images", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.app_state == "plan_execution":

    st.title("Analysis Plan Execution")

    plan_jsons = [
        h["analysis_plan"] for h in st.session_state.updated_hypotheses["hypotheses"]
    ]

    for i, raw_plan in enumerate(plan_jsons):
        plan  = json.loads(raw_plan)
        title = plan["analyses"][0]["title"]
        steps = plan["analyses"][0]["steps"]

        # ONE expander per hypothesis
        with st.expander(f"Hypothesis {i + 1}: {title}", expanded=False):

            # ①  show plan outline
            st.subheader("Steps")
            for s in steps:
                st.markdown(f"- {s['step']}")

            # ②  previous chat
            for msg in st.session_state.updated_hypotheses["hypotheses"][i][
                "plan_execution_chat_history"
            ]:
                with st.chat_message(msg["role"]):
                    for item in msg["items"]:
                        if item["type"] == "code_input":
                            st.code(item["content"])
                        elif item["type"] == "code_output":
                            st.code(item["content"])
                        elif item["type"] == "image":
                            for img in item["content"]:
                                st.markdown(img, unsafe_allow_html=True)
                        elif item["type"] == "text":
                            st.markdown(item["content"])
                        else:
                            raise ValueError(f"Unknown type: {item['type']}")

            prompt       = st.chat_input(
                "Let’s discuss the plan or execute it …",
                key=f"chat_input_plan_{i}",
            )

            if st.session_state.updated_hypotheses["hypotheses"][i]["plan_execution_chat_history"] == []:
                run_analysis = st.button("Run the analysis", key=f"run_analysis_{i}")
            else:
                run_analysis = st.button(
                    "Run the analysis again", key=f"run_analysis_{i}"
                )

            # kick things off
            if run_analysis or prompt:

                instructions = prompt if prompt else step_execution_instructions

                # save user message to the assistant’s thread
                client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content=f"\n\nThe analysis plan:\n{plan}\n",
                )

                # placeholders that live-update inside THIS expander
                container              = st.container()
                code_hdr_pl            = container.empty()
                code_pl                = container.empty()
                result_hdr_pl          = container.empty()
                result_pl              = container.empty()
                text_pl                = container.empty()

                assistant_output = []  # ← our running transcript

                # helper to make sure we have a “code_input” slot
                def ensure_code_slot():
                    if (
                        not assistant_output
                        or assistant_output[-1]["type"] != "code_input"
                    ):
                        assistant_output.append({"type": "code_input", "content": ""})

                # helper to make sure we have a “text” slot
                def ensure_text_slot():
                    if (
                        not assistant_output
                        or assistant_output[-1]["type"] != "text"
                    ):
                        assistant_output.append({"type": "text", "content": ""})

                # run the assistant and stream events
                stream = client.beta.threads.runs.create(
                    thread_id    = st.session_state.thread_id,
                    assistant_id = analysis_assistant.id,
                    instructions = instructions,
                    tool_choice  = {"type": "code_interpreter"},
                    stream       = True,
                )

                # ───────── STREAM HANDLER ─────────
                for event in stream:

                    # 1️⃣  Tool-calls step starts  → header “Writing code…”
                    if isinstance(event, ThreadRunStepCreated):
                        sd = event.data.step_details
                        if getattr(sd, "tool_calls", None):
                            ensure_code_slot()
                            code_hdr_pl.markdown("**Writing code ⏳ …**")

                    # 2️⃣  incremental code deltas
                    elif isinstance(event, ThreadRunStepDelta):
                        sd = event.data.delta.step_details
                        tc = getattr(sd, "tool_calls", None)
                        if tc and tc[0].code_interpreter:
                            delta = tc[0].code_interpreter.input or ""
                            if delta:
                                ensure_code_slot()
                                assistant_output[-1]["content"] += delta
                                code_pl.code(assistant_output[-1]["content"])

                    # 3️⃣  tool-calls step completed  → show results
                    elif isinstance(event, ThreadRunStepCompleted):
                        sd = event.data.step_details
                        tc = getattr(sd, "tool_calls", None)
                        if not tc:
                            continue
                        ci = tc[0].code_interpreter
                        outputs = ci.outputs or []
                        if not outputs:
                            continue

                        result_hdr_pl.markdown("#### Results")
                        
                        for out in outputs:
                            
                            if isinstance(out, CodeInterpreterOutputLogs):
                                logs = out.logs
                                assistant_output.append(
                                    {"type": "code_output", "content": logs}
                                )
                                result_pl.code(logs)

                            # (b) image(s)
                            elif isinstance(out, CodeInterpreterOutputImage):
                                fid  = out.image.file_id
                                data = client.files.content(fid).read()

                                # save
                                path = f"images/{fid}.png"
                                with open(path, "wb") as fp:
                                    fp.write(data)

                                # embed
                                b64  = base64.b64encode(data).decode("utf-8")
                                html = (
                                    f'<p align="center"><img src="data:image/png;base64,{b64}" '
                                    f'width="600"></p>'
                                )
                                result_pl.markdown(html, unsafe_allow_html=True)
                                img_html = html

                                if not assistant_output or assistant_output[-1]["type"] != "image":
                                    assistant_output.append({"type": "image", "content": []})

                                assistant_output[-1]["content"].append(img_html)

                    # 4️⃣  assistant plain-text message starts
                    elif isinstance(event, ThreadMessageCreated):
                        ensure_text_slot()

                    # 5️⃣  assistant plain-text deltas
                    elif isinstance(event, ThreadMessageDelta):
                        blk = event.data.delta.content[0]
                        if isinstance(blk, TextDeltaBlock):
                            ensure_text_slot()
                            assistant_output[-1]["content"] += blk.text.value
                            text_pl.markdown(assistant_output[-1]["content"])
                # ───── END STREAM HANDLER ────────

                # persist chat in session_state
                st.session_state.updated_hypotheses["hypotheses"][i][
                    "plan_execution_chat_history"
                ].append({"role": "assistant", "items": assistant_output})

                # client.beta.threads.messages.create(
                #     thread_id=st.session_state.thread_id,
                #     role="user",
                #     content=json.dumps(assistant_output, ensure_ascii=False),
                # )

                st.rerun()



################################################################
############ STAGE V: SUMMARY AND REPORT #######################
################################################################

# if st.session_state.app_state == "summary":