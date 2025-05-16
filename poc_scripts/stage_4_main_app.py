from __future__ import annotations
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import pandas as pd
from pydantic import BaseModel, RootModel
from typing import Optional, List, Dict
import ast

import json
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

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

from typing import List, Dict, Any, Optional

import base64
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI
from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta,
)
from openai.types.beta.threads.text_delta_block import TextDeltaBlock
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs,
)

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

## Set the page layout
st.set_page_config(page_title="Research assistant", 
                   page_icon=":tada:",
                   layout="wide")


data_summary_instructions = """
Task: Run Python code to read the provided files to summarize the dataset by analyzing its columns.
Extract and list all column names.
Always analyze the entire dataset, this is very important.
For each column:
- Provide column name.
- Infer a human-readable description of what the column likely represents.
- Identify the data type (e.g., categorical, numeric, text, date).
- Count the number of unique values.
"""

processing_files_instruction = """
## Task 
Based on the provided data summary, perform a critical analysis of each given hypothesis, and under the `hypothesis_refined_with_data_text` key do the following:
1. Assess Testability: Determine whether each hypothesis can be tested using the provided data. Justify your reasoning by referencing the relevant variables and their formats.
2. Identify Issues: Highlight any conceptual, statistical, or practical issues that may hinder testing — e.g., vague metrics, missing data, confounding variables, or unclear expected effects.
3. Refine Hypotheses: Suggest a clear and testable version of each hypothesis. 
4. Each refined hypothesis should: 
- Be logically structured and grounded in the data.
- Include a specific metric to be analyzed.
- Indicate the direction or nature of the expected change (e.g., increase/decrease, positive/negative correlation).
- Refined hypotheses should be framed in a way that is statistically testable.
5 Support with External Knowledge: If needed, search the web or draw from scientific literature to refine or inform the hypotheses.

## Instructions
- Under the `hypothesis_refined_with_data_text` write the more extended version of the hypotheses refinements.
- Under the `refined_hypothesis_text` field of your response always write a short, latest version of the refined hypothesis.
"""

refinig_instructions = """
## Role
You are an expert in ecological research and hypothesis development.
You have access to the dataset description provided by the user.
You can access internet resources to find up-to-date ecological research or contextual knowledge.
You are given a hypothesis that have been generated based on the dataset.

## Task
Your task is to help refine the hypotheses provided by the user based also on the user input.
Search the web for current reaserch related to the provided hypotheses.  WIKIPEDIA IS NOT A RELIABLE SOURCE!

## Instructions:
Important Constraints:
- Do not respond to any questions unrelated to the provided hypotheses.
- Use domain knowledge and data-driven reasoning to ensure each refined hypothesis is grounded in ecological theory and evidence.

For each hypothesis under the key `hypothesis_refined_with_data_text`:
1. Evaluate whether:
- It aligns with ecological theory or known patterns (search the web, WIKIPEDIA IS NOT A RELIABLE SOURCE!).
- Can be tested using the available data (based on variable types, structure, and coverage).
- If necessary, search the web for up-to-date ecological research or contextual knowledge to inform the refinement process.
- WIKIPEDIA IS NOT A RELIABLE SOURCE!
- Can it be tested? (Yes/No with explanation)
- Issues or concerns with the hypothesis (if any).
- Refined Hypothesis.
- Supporting context (optional, if external sources were used).

2. Suggest a refined version that:
- Clearly defines the expected relationship or effect.
- ALWAYS includes specific variables or metrics from the dataset. THIS IS VERY IMPORTANT!.
- Is phrased in a way that is statistically testable.

3. Under the `refined_hypothesis_text` field of your response always write a short, 
  latest version of the refined hypothesis.
"""

refining_chat_response_instructions = """
## Role
You are a seassoned expert in ecological research and hypothesis development. 
You are helping reserchers in their work by providing them assistance in hypotheses development.

## Task
You have access to the dataset description provided by the user.
You can access internet resources to find up-to-date ecological research or contextual knowledge.
You are given a hypothesis that have been generated based on the dataset.
Respond to the user query. 
When asked for your (assistant) response, 
search the web if you need current research context and provide references to your web searches.
WIKIPEDIA IS NOT A RELIABLE SOURCE!
In the `refined_hypothesis_text` field of your response always write a short, latest version of the refined hypothesis. 
"""

analyses_step_generation_instructions = """
## Role
- You are an expert in ecological research and statistical analysis**, with proficiency in **Python**.
- You must apply the **highest quality statistical methods and approaches** in data analysis.
- Your suggestions should be based on **best practices in ecological data analysis**.
- Your role is to **provide guidance, suggestions, and recommendations** within your area of expertise.
- Students will seek your assistance with **data analysis, interpretation, and statistical methods**.
- Since students have **limited statistical knowledge**, your responses should be **simple and precise**.
- Students also have **limited programming experience**, so provide **clear and detailed instructions**.

## Task
You have to generate an analysis plan for the provided hypothesis that can be tested on users dataset, for which a summary is provided.

## Instructions
- As the `assistant_chat_response`, generate a plan that is readable for the user contains explanations and motivations for the methods used.
- Keep a simpler version of the plan with clear and programmatically executable steps as `current_execution_plan` for further execution.
"""

analyses_step_chat_instructions = """
## Role
- You are an expert in ecological research and statistical analysis**, with proficiency in **Python**.
- You must apply the **highest quality statistical methods and approaches** in data analysis.
- Your suggestions should be based on **best practices in ecological data analysis**.
- Your role is to **provide guidance, suggestions, and recommendations** within your area of expertise.
- Students will seek your assistance with **data analysis, interpretation, and statistical methods**.
- Since students have **limited statistical knowledge**, your responses should be **simple and precise**.
- Students also have **limited programming experience**, so provide **clear and detailed instructions**.

## Task
You have to respond to a user querry about the analysis plan. 
Be profesional and provide best answer possible.
Search the web if necessary for the best and latest analytical tools.
Be encouraging, and suggest best solutions.

## Instructions
- As the `assistant_chat_response`, generate a plan that is readable for the user contains explanations and motivations for the methods used.
- Keep a simpler version of the plan with clear and programmatically executable steps as `current_execution_plan` for further execution.
"""

step_execution_assistant_instructions = """
## Role
You are an expert in ecological research and statistical analysis in Python. 
## Task
- execute the analysis plan provided by the user STEP BY STEP. 
- Write code in Python for each step to of the analysis plan from the beginning to the end.
- execute code, write description and short summary forr all of the steps.
"""

step_execution_chat_assistant_instructions = """
## Role
You are an expert in ecological research and statistical analysis in Python. 
## Task
- respond to the users queries about the elements of the analysis execution.
- write Python code as a response to the user query.
- execute code, write description and short summary as a response to the user query.
"""

step_execution_instructions = """
## Role
- You are an expert in ecological research and statistical analysis**, with proficiency in **Python**.
- You must apply the **highest quality statistical methods and approaches** in data analysis.
- Your suggestions should be based on **best practices in ecological data analysis**.
- Always try to **provide guidance, suggestions, and recommendations** within your area of expertise.
- Students will seek your assistance with **implementing steps of data analysis plan, interpretation, and statistical methods**.
- Since students have **limited statistical knowledge**, your responses should be **simple and precise**.
- Students also have **limited programming experience**, so provide **clear and detailed step by step implementations**

## Task
Execute in code every step of the analysis plan.
"""

step_execution_chat_instructions = """
Respond to the user prompt and refine parts of the provided analysis execution.
"""


STAGE_INFO = {
    "upload": "#### Stage I: Upload Files\n\n **Upload a CSV dataset and a TXT file containing your initial hypotheses.**\n\nOnce both are uploaded, the app automatically advances.\n\nFiles are held in `st.session_state`; the CSV preview is displayed with `st.dataframe()` so you can verify the data.",
    "processing":"### Stage II: Processing Files\n\n Great! You have your files uploaded.\n\nNow the app summarizes your dataset and rewrites each raw hypothesis into a clear, testable statement.\n\nYou’ll review them next.\n\nA GPT‑4o Data Summarizer assistant analyzes the CSV, then another GPT‑4o call refines the hypotheses using that summary; results are cached for later stages.",
    "hypotheses_manager": {
        "title": "3 · Hypotheses Manager",
        "description": (
            "Chat with the assistant to fine‑tune each hypothesis, then click "
            "Accept ✔️ to lock it in. All must be accepted before continuing."
        ),
        "how_it_works": (
            "Each hypothesis maintains its own chat history. Acceptance adds a "
            "final_hypothesis field, gating progression."
        ),
    },
    "plan_manager": {
        "title": "4 · Analysis Plan Manager",
        "description": (
            "For every accepted hypothesis, the assistant drafts a numbered "
            "analysis plan. Request revisions until it's perfect, then approve."
        ),
        "how_it_works": (
            "GPT-4o generates JSON-structured plans; the app validates the JSON "
            "before allowing approval."
        ),
    },
    "plan_execution": {
        "title": "5 · Plan Execution",
        "description": (
            "The assistant runs each approved plan in Python, streaming code, "
            "logs, and visuals live. You can pause, discuss, or rerun analyses."
        ),
        "how_it_works": (
            "A code-interpreter assistant streams run events. The app captures "
            "code inputs, outputs, and generated images, storing them for replay."
        ),
    },
    "report_generation": {
        "title": "6 · Scientific Report",
        "description": (
            "The assistant interprets results, searches recent ecological "
            "literature, and produces a full scientific report you can download."
        ),
        "how_it_works": (
            "The Report Generation assistant consolidates analysis outputs, "
            "performs web_search_preview calls for citations, writes an IMRaD "
            "report, and offers it as a Markdown download."
        ),
    },
}


# Response formats for ASSISTANTS
class DataSummary(BaseModel):
    column_name: str
    description: str
    type: str
    unique_value_count: int


class DatasetSummary(BaseModel):
    columns: Dict[str, DataSummary]


schema_payload = {
    "name": "summary_schema",
    "schema": DatasetSummary.model_json_schema()
}

response_format={
    "type": "json_schema",
    "json_schema": schema_payload
}


# Response formats for CHAT RESPONSES.
hypotheses_schema = {
            "format": {
                "type": "json_schema",
                "name": "hypotheses",
                "schema": {
                    "type": "object",
                    "properties": {
                        "assistant_response": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "hypothesis_refined_with_data_text": {"type": "string"},
                                    "refined_hypothesis_text": {"type": "string"},
                                },
                                "required": ["title", 
                                             "hypothesis_refined_with_data_text",
                                             "refined_hypothesis_text"],
                                "additionalProperties": False
                            }
                        },
                        "refined_hypothesis_text": {"type": "string"}
                    },
                    "required": ["assistant_response", "refined_hypothesis_text"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }


hyp_refining_chat_response_schema = {
    "format": {
        "type": "json_schema",
        "name": "hypothesis",
        "schema": {
            "type": "object",
            "properties": {
                "title": { "type": "string" },
                "assistant_response": { "type": "string" },
                "refined_hypothesis_text": { "type": "string" }
            },
            "required": [
                "title",
                "assistant_response",
                "refined_hypothesis_text"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

# Schema that works for plan generation and for the chat responses as well (line 936)
plan_generation_response_schema = {
    "format": {
        "type": "json_schema",
        "name": "plan_generation_response",
        "schema": {
            "type": "object",
            "properties": {"assistant_response":{"type": "string"}, # optional steps
                           "current_plan_execution":{"type": "string"}},
            "required": ["assistant_response","current_plan_execution"],
            "additionalProperties": False
        },
        "strict": True        
    }
}

# plan_generation_chat_response_schema = {"format": {"type": "text"}}


# ASSISTANTS
data_summary_assistant = client.beta.assistants.create(
    name="Data‑summarising Assistant",
    model="gpt-4o-2024-08-06",
    temperature=0,
    instructions=data_summary_instructions,
    tools=[{"type": "code_interpreter"}],
)


analysis_assistant = client.beta.assistants.create(
    name="Analysis Assistant",
    temperature=0,
    instructions=step_execution_assistant_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
    )


analysis_chat_assistant = client.beta.assistants.create(
    name="Analysis Assistant",
    temperature=0,
    instructions=step_execution_chat_assistant_instructions,
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

if "images" not in st.session_state:
    st.session_state.images = []

if "data_summary" not in st.session_state:
    st.session_state.data_summary = ""
if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = ""
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
if 'current_exec_idx' not in st.session_state:
    st.session_state['current_exec_idx'] = 0


def render_hypothesis_md(hyp: dict) -> str:
    """Return a markdown block for a single refined hypothesis."""
    md = [f"### {hyp['title']}"]

    # Rationale steps
    if hyp.get("steps"):
        md.append("**Rationale**:")
        for i, s in enumerate(hyp["steps"], start=1):
            md.append(f"{i}. {s['step']}")

    # Final refined hypothesis (if filled later in workflow)
    if hyp.get("final_hypothesis"):
        md.append("\n> **Final refined hypothesis**:\n>")
        md.append(f"> {hyp['final_hypothesis']}")

    return "\n".join(md)


def init_state():
    defaults = dict(
        app_state="upload",         
        data_uploaded=False,
        hypotheses_uploaded=False,
        processing_done=False,
        files={},                    
        file_ids=[],
        hypotheses="",
        data_summary="",
        updated_hypotheses={},
        thread_id="",
        selected_hypothesis=0,
        analysis_plan_chat_history=[],
        plan_execution_chat_history=[]
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


st.markdown(
    """
    <style>
        /* Enabled button  = green */
        div.stButton > button:enabled {
            background-color: #28a745 !important;   /* green  */
            color: white            !important;
        }

        /* Disabled button = grey (optional, makes the state obvious) */
        div.stButton > button:disabled {
            background-color: #d0d0d0 !important;
            color: #808080            !important;
            cursor: not-allowed       !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hypothesis_md(hyp: dict) -> str:
    """Return a markdown block for a single refined hypothesis."""
    md = [f"### {hyp['title']}"]

    if hyp.get("hypothesis_refined_with_data_text"):
        # md.append("\n> **Refined hypothesis**:\n>")
        md.append(f"> {hyp['hypothesis_refined_with_data_text']}")

    return "\n".join(md)


def format_initial_assistant_msg(hyp: dict) -> str:
    """Return formatted markdown for the assistant seed message."""
    return f"**Refined hypothesis:** {hyp['title']}\n\n{hyp['hypothesis_refined_with_data_text']}"


def stream_data_summary(client: OpenAI):
                 
    for name, f in st.session_state.files.items():
        print(f"Uploading {name} …")
        file_obj = client.files.create(file=f, purpose="assistants")
        st.session_state.file_ids.append(file_obj.id)

    print(f"File IDs: {st.session_state.file_ids}")

    thread_id = st.session_state.thread_id

    print(f"Thread ID: {thread_id}")
    
    client.beta.threads.update(
        thread_id=thread_id,
        tool_resources={"code_interpreter": {"file_ids": st.session_state.file_ids}},
    )

    print(f"Files added to thread {thread_id}")

    container      = st.container()
    code_hdr_pl    = container.empty()
    code_pl        = container.empty()
    result_hdr_pl  = container.empty()
    result_pl      = container.empty()
    json_hdr_pl    = container.empty()
    json_pl        = container.empty()
    text_pl        = container.empty()

    assistant_items: list[dict] = []

    def ensure_slot(tp: str):
        if not assistant_items or assistant_items[-1]["type"] != tp:
            assistant_items.append({"type": tp, "content": ""})

    stream = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=data_summary_assistant.id,
        response_format=response_format,
        stream=True
    )

    for event in stream:
        # ---- code‑interpreter life‑cycle -----------------------------------
        if isinstance(event, ThreadRunStepCreated):
            if getattr(event.data.step_details, "tool_calls", None):
                ensure_slot("code_input")
                code_hdr_pl.markdown("**Running code ⏳ …**")

        elif isinstance(event, ThreadRunStepDelta):
            tc = getattr(event.data.delta.step_details, "tool_calls", None)
            if tc and tc[0].code_interpreter:
                delta = tc[0].code_interpreter.input or ""
                if delta:
                    ensure_slot("code_input")
                    assistant_items[-1]["content"] += delta
                    code_pl.code(assistant_items[-1]["content"], language="python")

        elif isinstance(event, ThreadRunStepCompleted):
            tc = getattr(event.data.step_details, "tool_calls", None)
            if not tc:
                continue
            outputs = tc[0].code_interpreter.outputs or []
            if outputs:
                result_hdr_pl.markdown("#### Code‑interpreter output")
            for out in outputs:
                if isinstance(out, CodeInterpreterOutputLogs):
                    ensure_slot("code_output")
                    assistant_items[-1]["content"] += out.logs
                    result_pl.code(out.logs)
                elif isinstance(out, CodeInterpreterOutputImage):
                    fid  = out.image.file_id
                    data = client.files.content(fid).read()
                    b64  = base64.b64encode(data).decode()
                    html = f'<p align="center"><img src="data:image/png;base64,{b64}" width="600"></p>'
                    ensure_slot("image")
                    assistant_items[-1]["content"] += html
                    result_pl.markdown(html, unsafe_allow_html=True)

        # ---- assistant's JSON answer (delta‑streamed) ----------------------
        elif isinstance(event, ThreadMessageCreated):
            ensure_slot("json")               # we know the next message is JSON
            json_hdr_pl.markdown("#### Column summary (streaming)")

        elif isinstance(event, ThreadMessageDelta):
            blk = event.data.delta.content[0]
            if isinstance(blk, TextDeltaBlock):
                ensure_slot("json")
                assistant_items[-1]["content"] += blk.text.value
                # prettify incremental JSON (optional)
                json_pl.markdown(f"```json\n{assistant_items[-1]['content']}\n```")

    try:
        messages = client.beta.threads.messages.list(
            thread_id=thread_id, order="desc"
        ).data[0]

        # print(f"Here sould be JSON conforming to the schema:\n\n{messages.content[0].text.value}")


        summary_dict = json.loads(messages.content[0].text.value)
    except Exception as e:
        st.error(f"❌ Could not parse JSON: {e}")
        return

    st.session_state.data_summary = summary_dict


# SIDEBAR – STAGE 1 UPLOADS

if st.session_state.app_state in {"upload", "processing"}:

    if st.session_state.app_state == "upload":
        st.markdown(STAGE_INFO["upload"])

    with st.sidebar:
        st.header("📂 Upload files")

        # ── CSV
        csv_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
        if csv_file:
            st.toast("CSV file uploaded!", icon="🎉")
            df_preview = pd.read_csv(csv_file)
            st.session_state.files[csv_file.name] = csv_file
            st.session_state.data_uploaded = True
            with st.expander("Data preview"):
                st.dataframe(df_preview.head())

        # ── TXT
        txt_file = st.file_uploader("Choose a TXT file", type="txt", key="txt_uploader")
        if txt_file:
            st.toast("TXT file uploaded!", icon="📝")
            st.session_state.hypotheses = txt_file.read().decode("utf‑8")
            st.session_state.hypotheses_uploaded = True
            with st.expander("Hypotheses preview"):
                st.text_area("File content", st.session_state.hypotheses, height=180)

        # Auto‑advance once both files are uploaded
        if (
            st.session_state.data_uploaded
            and st.session_state.hypotheses_uploaded
            and st.session_state.app_state == "upload"
        ):
            st.session_state.app_state = "processing"
            st.rerun()













# ────────────────────────────────────────────────────────────────────────────────
# MAIN AREA – STAGE 2  (PROCESSING)
# ────────────────────────────────────────────────────────────────────────────────


if st.session_state.app_state == "processing":
    
    st.markdown(STAGE_INFO["processing"])

    if st.session_state.get("data_summary"):
        with st.expander("📊 Data summary", expanded=False):
            meta = st.session_state.data_summary
            st.markdown("#### Dataset summary")
            # Chack if the response 
            for col, m in meta["columns"].items():
                st.markdown(f"##### {col}\n*Description:* {m['description']}\n\n*Type:* {m['type']}.\n\n*Unique values:* {m['unique_value_count']}\n")
            # st.markdown(st.session_state.data_summary)

    if st.session_state.get("updated_hypotheses"):
        st.subheader("Refined hypotheses")
        for hyp in st.session_state.updated_hypotheses["assistant_response"]:
            with st.expander(hyp['title'], expanded=False):
                st.markdown(render_hypothesis_md(hyp))

    ### Footer
    footer = st.container()
    col1, col2 = footer.columns(2)

    with col1:
        process_click = st.button(
            "🚀 Process Files",
            disabled=st.session_state.processing_done,
            key="process_files"
        )
        
    if process_click:
        
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

        # Stream data summary creation
        stream_data_summary(client)

        st.session_state.processing_done = True
        
        with st.expander("📊 Data summary", expanded=False):
            st.json(st.session_state.data_summary)

        # Refining prompt stage
        refine_prompt = (
            f"Data summary: {st.session_state.data_summary}\n\n"
            f"Hypotheses: {st.session_state.hypotheses}\n\n"
            f"{processing_files_instruction}\n"
        )

        response = client.responses.create(
            model="gpt-4o",
            instructions=refinig_instructions,
            input=[{"role": "user", "content": refine_prompt}],
            tools=[{"type": "web_search_preview"}],
            text=hypotheses_schema,
        )

        # The hypotheses are being updated here
        st.session_state.updated_hypotheses = json.loads(response.output_text)

        for hyp in st.session_state.updated_hypotheses["assistant_response"]:
            pretty_msg = format_initial_assistant_msg(hyp)
            hyp["chat_history"] = [{"role": "assistant", "content": pretty_msg}]
            hyp["final_hypothesis"] = []


        st.session_state.processing_done = True
        st.success("Processing complete!", icon="✅")
        st.rerun()

    # Show refined hypotheses summary (for preview)

    # MOVE TO NEXT STEP BUTTON
    with col2:
        next_click = st.button(
            "➡️ Move to next step",
            disabled=not st.session_state.processing_done,
            key="next_step",
        )
        if next_click:
            st.session_state.app_state = "hypotheses_manager"
            st.rerun()



if st.session_state.app_state == "hypotheses_manager":
    # ── SIDEBAR: list of hypotheses --------------------------------------------
    with st.sidebar:
        st.header("📑 Hypotheses")

        for idx, hyp in enumerate(st.session_state.updated_hypotheses["assistant_response"]):
            with st.expander(hyp["title"], expanded=False):
                
                if hyp["final_hypothesis"]:
                    st.markdown(f"> {hyp['final_hypothesis']}")

                else:
                    st.markdown(f"> {hyp['hypothesis_refined_with_data_text']}")
                
                if st.button("✏️ Edit", key=f"select_{idx}"):
                    st.session_state.selected_hypothesis = idx
                    st.rerun()

        # for idx, hyp in enumerate(st.session_state.updated_hypotheses["hypotheses"]):
        #     with st.expander(hyp["title"], expanded=False):
        #         # show either final hypothesis or rationale steps
        #         if hyp["final_hypothesis"]:
        #             st.markdown(f"> {hyp['final_hypothesis']}")
        #         else:
        #             for j, step in enumerate(hyp["steps"], start=1):
        #                 st.markdown(f"{j}. {step['step']}")

        #         if st.button("✏️ Edit", key=f"select_{idx}"):
        #             st.session_state.selected_hypothesis = idx
        #             st.rerun()

    # ── MAIN CANVAS: chat & accept button -------------------------------------
    sel_idx = st.session_state.selected_hypothesis
    sel_hyp = st.session_state.updated_hypotheses["assistant_response"][sel_idx]

    # import pprint
    # pprint.pprint(f"\n\nSELECTED HYPOTHESIS FROM THE UPDATED HYPOTHESES:\n\n{sel_hyp}")

    # st.subheader(STAGE_INFO["hypotheses_manager"]["title"])
    # st.write(STAGE_INFO["hypotheses_manager"]["description"])
    # st.write(STAGE_INFO["hypotheses_manager"]["how_it_works"])

    st.subheader(f"🗣️ Discussion – {sel_hyp['title']}")

    # display chat history
    for msg in sel_hyp["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat input
    user_prompt = st.chat_input("Refine this hypothesis further …", key=f"chat_input_{sel_idx}")

    if user_prompt:
        sel_hyp["chat_history"].append({"role": "user", "content": user_prompt})

        response_input_history = [{
            k: v for k, v in d.items() if k != "refined_hypothesis_text"} for d in sel_hyp["chat_history"]
            ]

        with st.spinner("Thinking …"):
            response = client.responses.create(
                model="gpt-4o",
                instructions=refining_chat_response_instructions,
                input= [{
                    "role": "user", 
                    "content": f"Here is the summary of the data: {st.session_state.data_summary}"
                }] + response_input_history,
                tools=[{"type": "web_search_preview"}],
                text = hyp_refining_chat_response_schema,
            )
        
        response_json = json.loads(response.output_text)

        # print(f"\n\nTHE RESPONSE JSON:\n\n{response_json}")

        sel_hyp["chat_history"].append(
            {"role": "assistant", 
             "content": response_json["assistant_response"],
             "refined_hypothesis_text": response_json["refined_hypothesis_text"]}
        )
        st.rerun()

    # ACCEPT BUTTON
    acc_disabled = bool(sel_hyp["final_hypothesis"])
    
    if st.button("✅ Accept refined hypothesis", disabled=acc_disabled, key="accept"):
        if len(sel_hyp["chat_history"]) > 1:
            sel_hyp["final_hypothesis"] = sel_hyp["chat_history"][-1]["refined_hypothesis_text"]
        else:
            sel_hyp["final_hypothesis"] = sel_hyp["refined_hypothesis_text"]
        st.success("Hypothesis accepted!")
        st.rerun()

    # ── AUTO‑ADVANCE -----------------------------------------------------------
    if all(h["final_hypothesis"] for h in st.session_state.updated_hypotheses["assistant_response"]):
        st.session_state.app_state = "plan_manager"  # next stage placeholder
        st.rerun()














# STAGE 3 ANALYSIS PLAN MANAGER

def pretty_markdown_plan(raw_json: str) -> str:
    """Convert the assistant‑returned JSON (analyses → steps) into Markdown."""
    try:
        data = json.loads(raw_json)
        md_blocks = []
        for ana in data.get("analyses", []):
            md_blocks.append(f"### {ana['title']}\n")
            for idx, step in enumerate(ana["steps"], 1):
                md_blocks.append(f"{idx}. {step['step']}")
            md_blocks.append("\n")
        return "\n".join(md_blocks)
    except Exception:
        # fall back to raw text if parsing fails
        return raw_json

def ensure_plan_keys(h):
    h.setdefault("analysis_plan_chat_history", [])
    h.setdefault("analysis_plan", "")
    h.setdefault("analysis_plan_accepted", False)
    return h


def plan_manager(client: OpenAI):
    """Stage 3 - manage analysis - plan creation & approval."""
    
    if st.session_state.app_state != "plan_manager":
        return

    with st.sidebar:
        st.header("Accepted hypotheses")
        for idx, h in enumerate(st.session_state.updated_hypotheses["assistant_response"]):
            title = f"Hypothesis {idx+1}"
            with st.expander(title, expanded=False):
                st.markdown(h["final_hypothesis"], unsafe_allow_html=True)
                if st.button("✏️ Work on this", key=f"select_hypo_{idx}"):
                    st.session_state["current_hypothesis_idx"] = idx
                    st.rerun()

    # Which hypothesis is in focus?
    current = st.session_state.get("current_hypothesis_idx", 0)

    hypo_obj = ensure_plan_keys(
        st.session_state.updated_hypotheses["assistant_response"][current]
    )

    st.subheader(f"Analysis Plan Manager: Hypothesis {current+1}")
    st.markdown(hypo_obj["final_hypothesis"], unsafe_allow_html=True)

    # Plan generation / chat
    chat_hist = hypo_obj["analysis_plan_chat_history"]

    if not chat_hist:  # first‑time plan generation
        if st.button("Generate plan", key="generate_plan"):
            prompt = (
                f"Here is the data summary: {st.session_state.data_summary}\n\n"
                f"Here is the hypothesis: {hypo_obj['final_hypothesis']}"
            )

            prompt_str = "".join(prompt)

            chat_hist.append({"role": "user", "content": prompt_str})

            with st.spinner("Generating …"):
                resp = client.responses.create(
                    model="gpt-4o",
                    temperature=0,
                    instructions=analyses_step_generation_instructions,
                    input=prompt_str,
                    stream=False,
                    tools=[{"type": "web_search_preview"}],
                    text=plan_generation_response_schema,
                    store=False,
                )

            # print(f"\n\nResponse from the plan generation response:\n\n{resp}")

            chat_hist.append({"role": "assistant", "content": resp.output_text})

            st.rerun()

    if not hypo_obj["analysis_plan_accepted"]:
        
        # Show existing chat
        for m in chat_hist[1:]:
            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    st.markdown(
                        json.loads(m["content"])["assistant_response"], 
                        unsafe_allow_html=True
                    )
                elif m["role"] == "user":
                    st.write(m["content"])

        user_msg = st.chat_input("Refine this analysis plan …")

        if user_msg:
            chat_hist.append({"role": "user", "content": user_msg})
            
            with st.spinner("Thinking …"):
                resp = client.responses.create(
                    model="gpt-4o",
                    temperature=0,
                    instructions=analyses_step_chat_instructions,
                    input=chat_hist,
                    stream=False,
                    tools=[{"type": "web_search_preview"}],
                    text=plan_generation_response_schema,
                    store=False,
                )
            chat_hist.append({"role": "assistant", "content": resp.output_text})
            st.rerun()

        if chat_hist:
            if st.button("✅ Accept this plan", key="accept_plan"):
                hypo_obj["analysis_plan"] = chat_hist[-1]["content"]
                hypo_obj["analysis_plan_accepted"] = True
                st.rerun()

    if hypo_obj["analysis_plan_accepted"]:
        st.success("Plan accepted")

        plan_json = json.loads(hypo_obj["analysis_plan"])

        st.markdown(plan_json["assistant_response"], unsafe_allow_html=True)

    all_ready = all(
        h.get("analysis_plan") and h.get("analysis_plan_accepted")
        for h in st.session_state.updated_hypotheses["assistant_response"]
    )

    if all_ready:

        # print(st.session_state.updated_hypotheses["assistant_response"])

        if st.button("➡️ Move to plan execution stage"):
            st.session_state.app_state = "plan_execution"
            st.rerun()

if st.session_state.app_state == "plan_manager":
    plan_manager(client)






















IMG_DIR = Path("images"); IMG_DIR.mkdir(exist_ok=True)
JSON_RE  = re.compile(r"\{[\s\S]*?\}")
STEP_RE  = re.compile(r"^(?:\d+\.\s+|[-*+]\s+)(.+)")

def extract_json_fragment(text: str) -> Optional[str]:
    m = JSON_RE.search(text)
    return m.group(0) if m else None

def _mk_fallback_plan(text: str) -> Dict[str, Any]:
    """Convert a loose Markdown plan → canonical dict."""
    lines   = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title   = lines[0].lstrip("# ") if lines else "Analysis Plan"
    steps   = []
    for ln in lines[1:]:
        m = STEP_RE.match(ln)
        if m:
            steps.append({"step": m.group(1).strip()})
    if not steps:  # fall back to one‑chunk step
        steps = [{"step": text.strip()}]
    return {"analyses": [{"title": title, "steps": steps}]}

def safe_load_plan(raw: Any) -> Optional[Dict[str, Any]]:
    """Return plan dict from dict / JSON / python literal / markdown."""
    if isinstance(raw, dict):
        return raw
    if not raw:  # empty / None
        return None

    if isinstance(raw, str):
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.lstrip("` pythonjson").rstrip("`").strip()

        # 1️⃣ json.loads with double quotes
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # 2️⃣ ast.literal_eval for single‑quoted dicts
            try:
                obj = ast.literal_eval(txt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            # 3️⃣ fragment inside markdown
            frag = extract_json_fragment(txt)
            if frag:
                try:
                    return json.loads(frag)
                except json.JSONDecodeError:
                    pass
            # 4️⃣ fallback – build from markdown bullets
            return _mk_fallback_plan(txt)
    return None

def ensure_execution_keys(h):
    h.setdefault("plan_execution_chat_history", [])
    return h


def plan_execution(client: OpenAI):
    if st.session_state.app_state != "plan_execution":
        return

    st.title("Analysis Plan Execution")

    # ------------------------------------------------------------------
    # Sidebar – hypotheses
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Accepted hypotheses")
        for idx, h in enumerate(st.session_state.updated_hypotheses["assistant_response"]):
            title = f"Hypothesis {idx+1}"
            with st.expander(title, expanded=False):
                st.markdown(h["final_hypothesis"], unsafe_allow_html=True)
                if st.button("▶️ Run / review", key=f"select_exec_{idx}"):
                    st.session_state["current_exec_idx"] = idx
                    st.rerun()

    # Which hypothesis are we executing?
    current = st.session_state.get("current_exec_idx", 0)
    hypo_obj = ensure_execution_keys(
        st.session_state.updated_hypotheses["assistant_response"][current])

    plan_dict = safe_load_plan(hypo_obj["analysis_plan"])

    if not plan_dict:
        st.error("❌ Could not parse analysis plan JSON. Please regenerate the plan in the previous stage or ask the assistant to output valid JSON.")
        return
    
    # print(f"\n\nPLAN DICT\n\n:{plan_dict}")

    st.markdown(plan_dict['assistant_response'])
    st.markdown(plan_dict['current_plan_execution'])

    # Normalise stored plan to dict for future safety
    if isinstance(hypo_obj["analysis_plan"], str):
        hypo_obj["analysis_plan"] = plan_dict


    # Previous transcript ------------------------------------------------
    for msg in hypo_obj["plan_execution_chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                for item in msg["items"]:
                    if item["type"] == "code_input":
                        st.code(item["content"], language="python")
                    elif item["type"] == "code_output":
                        st.code(item["content"], language="text")
                    elif item["type"] == "image":
                        for html in item["content"]:
                            st.markdown(html, unsafe_allow_html=True)
                    elif item["type"] == "text":
                        st.markdown(item["content"], unsafe_allow_html=True)
            else:
                st.markdown(msg["content"], unsafe_allow_html=True)

    # Prompt & run button ------------------------------------------------
    user_prompt = st.chat_input(
        "Discuss the plan or ask to run specific steps …",
        key="exec_chat_input",
    )

    run_label = (
        "▶️ Run analysis" if not hypo_obj["plan_execution_chat_history"] else "🔄 Run analysis again"
    )
    run_analysis = st.button(run_label, key="run_analysis_btn")

    if run_analysis or user_prompt:
        # Choose assistant instructions
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=f"\n\nThe analysis plan:\n{json.dumps(plan_dict, indent=2)}",
        )

        # Live placeholders
        container      = st.container()
        code_hdr_pl    = container.empty()
        code_pl        = container.empty()
        result_hdr_pl  = container.empty()
        result_pl      = container.empty()
        text_pl        = container.empty()

        assistant_items: List[Dict[str, Any]] = []

        def ensure_slot(tp: str):
            if not assistant_items or assistant_items[-1]["type"] != tp:
                assistant_items.append({"type": tp, "content": "" if tp != "image" else []})

        if run_analysis:
            # Runs the initial analysis with specific instructions.
            stream = client.beta.threads.runs.create(
                thread_id    = st.session_state.thread_id,
                assistant_id = analysis_assistant.id,
                instructions = step_execution_instructions,
                tool_choice  = {"type": "code_interpreter"},
                stream       = True,
            )

        elif user_prompt:
            # Responds to user, focuses on refining parts of the plan.
            hypo_obj["plan_execution_chat_history"].append(
                {"role": "user", "content": user_prompt}
            )

            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=user_prompt,
            )

            stream = client.beta.threads.runs.create(
                thread_id    = st.session_state.thread_id,
                assistant_id = analysis_chat_assistant.id,
                instructions = step_execution_chat_instructions,
                tool_choice  = {"type": "code_interpreter"},
                stream       = True,
            )



        for event in stream:
            if isinstance(event, ThreadRunStepCreated):
                if getattr(event.data.step_details, "tool_calls", None):
                    ensure_slot("code_input")
                    code_hdr_pl.markdown("**Writing code ⏳ …**")

            elif isinstance(event, ThreadRunStepDelta):
                tc = getattr(event.data.delta.step_details, "tool_calls", None)
                if tc and tc[0].code_interpreter:
                    delta = tc[0].code_interpreter.input or ""
                    if delta:
                        ensure_slot("code_input")
                        assistant_items[-1]["content"] += delta
                        code_pl.code(assistant_items[-1]["content"], language="python")

            elif isinstance(event, ThreadRunStepCompleted):
                tc = getattr(event.data.step_details, "tool_calls", None)
                if not tc:
                    continue
                outputs = tc[0].code_interpreter.outputs or []
                if not outputs:
                    continue
                result_hdr_pl.markdown("#### Results")
                for out in outputs:
                    if isinstance(out, CodeInterpreterOutputLogs):
                        ensure_slot("code_output")
                        assistant_items[-1]["content"] += out.logs
                        result_pl.code(out.logs)
                    elif isinstance(out, CodeInterpreterOutputImage):
                        fid  = out.image.file_id
                        data = client.files.content(fid).read()
                        img_path = IMG_DIR / f"{fid}.png"
                        img_path.write_bytes(data)


                        # After saving the image data, add it to the thread
                        st.session_state.images.append({"image_path": img_path, "file_id": fid})

                        print(f"Image: {out.image.file_id} at {IMG_DIR / f'{fid}.png'} added to the thread {st.session_state.thread_id}.")

                        b64 = base64.b64encode(data).decode()
                        html = (
                            f'<p align="center"><img src="data:image/png;base64,{b64}" '
                            f'width="600"></p>'
                        )
                        ensure_slot("image")
                        assistant_items[-1]["content"].append(html)
                        result_pl.markdown(html, unsafe_allow_html=True)

            elif isinstance(event, ThreadMessageCreated):
                ensure_slot("text")

            elif isinstance(event, ThreadMessageDelta):
                blk = event.data.delta.content[0]
                if isinstance(blk, TextDeltaBlock):
                    ensure_slot("text")
                    assistant_items[-1]["content"] += blk.text.value
                    text_pl.markdown(assistant_items[-1]["content"], unsafe_allow_html=True)

        # After the run ends add the images to the thread
        for img in st.session_state["images"]:
            file = client.files.create(
                            file=open(img["image_path"], "rb"),
                            purpose="vision"
                        )

            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=[{"type": "image_file","image_file": {"file_id": file.id}}]
            )

        hypo_obj["plan_execution_chat_history"].append(
            {"role": "assistant", "items": assistant_items}
        )

        st.rerun()


if st.session_state.app_state == "plan_execution":
    plan_execution(client)
    # print(f"\n\nTHE ASSISTANT RESPONSE AFTER CODE EXECUTION:\n\n{st.session_state.updated_hypotheses['assistant_response'][0]['plan_execution_chat_history']}")
    if st.sidebar.button("➡️ Generate final report"):
        st.session_state.app_state = "report_generation"
        st.rerun()



# STAGE 4. REPORT GENERATION

report_generation_instructions = """
You are an expert ecological scientist and statistician.
Your task is to craft a peer‑reviewed‑quality report based on:
• The refined hypotheses tested;  
• The statistical results produced in the previous stage;  
• Any additional context you can gather from current literature.

**Report structure (Markdown):**
1. **Methodology** - one paragraph describing data sources, key variables, and
   statistical procedures actually executed (e.g., GLM, mixed-effects model,
   correlation analysis, etc.).  *Use past tense.*
2. **Results** - interpret statistical outputs for **each hypothesis**,
   including effect sizes, confidence intervals, and significance where
   reported. Embed any relevant numeric values (means, p-values, etc.).
   Include relevant images and write captions for them.
3. **Interpretations** - compare findings with recent studies retrieved via
   `web_search_preview`; highlight agreements, discrepancies, and plausible
   ecological mechanisms.
4  **Conclusion** - wrap-up of insights and recommendations for future work.

*Write in formal academic style, using citations like* “(Smith 2024)”, and provite DOI for each one.

If web search yields no directly relevant article, proceed without citation.
"""

report_asst = client.beta.assistants.create(
        name="Report Generation Assistant",
        model="gpt-4.1",
        temperature=0,
        instructions=report_generation_instructions,
        tools=[{"type": "code_interpreter"}],
    )


def build_report_prompt():
    report_prompt = []
    for idx, hyp in enumerate(st.session_state.updated_hypotheses['assistant_response']):
        for msg in hyp['plan_execution_chat_history']:
            # print(f"\n\nThe message:\n\n {msg.keys()}")
            if "items" in msg:
                for item in msg["items"]:
                    # Exclude outputs!!!
                    if item["type"] not in ("image", "code_output"):
                        report_prompt.append(item["content"])
            elif "content" in msg:
                report_prompt.append(msg["content"])
    return " ".join(report_prompt)


# def build_report_prompt():
#     print("Saving the whole output to assistant_response.json...")
#     with open("assistant_response.json", "w") as f:
#         json.dump(st.session_state.updated_hypotheses['assistant_response'], f, indent=4)

#     report_prompt = []

#     for idx, hyp in enumerate(st.session_state.updated_hypotheses['assistant_response']):
#         for msg in hyp['plan_execution_chat_history']:
#             if "items" in msg:
#                 for item in msg["items"]:
#                     content = item["content"]
#                     if isinstance(content, list):
#                         report_prompt.extend(str(x) for x in content)
#                     else:
#                         report_prompt.append(str(content))
#             if "content" in msg:
#                 if isinstance(msg["content"], list):
#                     report_prompt.extend(str(x) for x in msg["content"])
#                 else:
#                     report_prompt.append(str(msg["content"]))

#     return " ".join(report_prompt)










#-------------------------------------------------------------------------------------------




def report_generation(client: OpenAI):
    """Render the Report Generation stage and orchestrate the response call."""

    if st.session_state.app_state != "report_generation":
        return

    st.title("📄 Report Builder")

    # Sidebar – quick outline of accepted hypotheses
    with st.sidebar:
        st.header("Refined Initial Hypotheses")
        for idx, hyp in enumerate(st.session_state.updated_hypotheses["assistant_response"], 1):
            st.markdown(f"**H{idx}.** {hyp['title']}")

    # Button to trigger report generation
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
        st.session_state.report_markdown = ""

    # if st.button("📝 Generate full report", disabled=st.session_state.report_generated):
    if st.button("📝 Generate full report"):
        # Build a report that contains text, code, images and tables (Can I generate tables?)
        full_prompt = build_report_prompt()

        if "final_report" not in st.session_state:
                st.session_state["final_report"] = []

        with st.spinner("Synthesising report – this may take a minute …"):
            # stream = client.beta.threads.runs.create(
            #     thread_id=st.session_state.thread_id,
            #     assistant_id=report_asst.id,
            #     stream=True
            #     )

            response = client.responses.create(
                model="gpt-4.1",
                instructions=report_generation_instructions,
                input=[{"role": "user", "content": full_prompt}],
                tools=[{"type": "web_search_preview"}]
                )
            
            print(response.output_text)
            
            st.session_state["final_report"].append(response.output_text)
            st.session_state.report_generated = True

        st.rerun()

    # Display the generated report
    if st.session_state.report_generated:
        st.markdown(st.session_state.final_report[1], unsafe_allow_html=True)

        # Offer download as Markdown
        st.download_button(
            "⬇️ Download report (Markdown)",
            st.session_state.final_report[1],
            file_name="scientific_report.md",
            mime="text/markdown",
        )

        # Optionally, add a next‑steps button to reset or exit
        if st.button("🔄 Start new session"):
            st.session_state.clear()
            st.experimental_rerun()







# def report_generation(client: OpenAI):
#     """Render the Report Generation stage and orchestrate the response call."""

#     if st.session_state.app_state != "report_generation":
#         return

#     st.title("📄 Report Builder")

#     # Sidebar – quick outline of accepted hypotheses
#     with st.sidebar:
#         st.header("Refined Initial Hypotheses")
#         for idx, hyp in enumerate(st.session_state.updated_hypotheses["assistant_response"], 1):
#             st.markdown(f"**H{idx}.** {hyp['title']}")

#     # Button to trigger report generation
#     if "report_generated" not in st.session_state:
#         st.session_state.report_generated = False
#         st.session_state.report_markdown = ""

#     if st.button("📝 Generate full report", disabled=st.session_state.report_generated):
        
#         # Build a report that contains text, code, images and tables (Can I generate tables?)
#         full_prompt = build_report_prompt()

#         print(f"\n\nFULL PROMPT\n\n{full_prompt}")

#         with st.spinner("Synthesising report – this may take a minute …"):
#             stream = client.beta.threads.runs.create(
#                 thread_id=st.session_state.thread_id,
#                 assistant_id=report_asst.id,
#                 stream=True
#                 )
            
#         # Live placeholders
#         container      = st.container()
#         code_hdr_pl    = container.empty()
#         code_pl        = container.empty()
#         result_hdr_pl  = container.empty()
#         result_pl      = container.empty()
#         text_pl        = container.empty()

#         assistant_items: List[Dict[str, Any]] = []

#         def ensure_slot(tp: str):
#             if not assistant_items or assistant_items[-1]["type"] != tp:
#                 assistant_items.append({"type": tp, "content": "" if tp != "image" else []})

#         ## HANDLER
#         for event in stream:
#             if isinstance(event, ThreadRunStepCreated):
#                 if getattr(event.data.step_details, "tool_calls", None):
#                     ensure_slot("code_input")
#                     code_hdr_pl.markdown("**Writing code ⏳ …**")

#             elif isinstance(event, ThreadRunStepDelta):
#                 tc = getattr(event.data.delta.step_details, "tool_calls", None)
#                 if tc and tc[0].code_interpreter:
#                     delta = tc[0].code_interpreter.input or ""
#                     if delta:
#                         ensure_slot("code_input")
#                         assistant_items[-1]["content"] += delta
#                         code_pl.code(assistant_items[-1]["content"], language="python")

#             elif isinstance(event, ThreadRunStepCompleted):
#                 tc = getattr(event.data.step_details, "tool_calls", None)
#                 if not tc:
#                     continue
#                 outputs = tc[0].code_interpreter.outputs or []
#                 if not outputs:
#                     continue
#                 result_hdr_pl.markdown("#### Results")
#                 for out in outputs:
#                     if isinstance(out, CodeInterpreterOutputLogs):
#                         ensure_slot("code_output")
#                         assistant_items[-1]["content"] += out.logs
#                         result_pl.code(out.logs)
#                     elif isinstance(out, CodeInterpreterOutputImage):
#                         fid  = out.image.file_id
#                         data = client.files.content(fid).read()
                        
#                         img_path = IMG_DIR / f"{fid}.png"
#                         img_path.write_bytes(data)

#                         # After saving the image data, add it to the thread
#                         file = client.files.create(
#                             file=open(img_path, "rb"),
#                             purpose="vision"
#                         )

#                         print(f"Image: {out.image.file_id} at {IMG_DIR / f'{fid}.png'} added to the thread {st.session_state.thread_id}.")

#                         client.beta.threads.messages.create(
#                             thread_id=st.session_state.thread_id,
#                             role="user",
#                             content=[{"type": "image_file","image_file": {"file_id": file.id}}]
#                         )

                        

#                         b64 = base64.b64encode(data).decode()
#                         html = (
#                             f'<p align="center"><img src="data:image/png;base64,{b64}" '
#                             f'width="600"></p>'
#                         )
#                         ensure_slot("image")
#                         assistant_items[-1]["content"].append(html)
#                         result_pl.markdown(html, unsafe_allow_html=True)

#             elif isinstance(event, ThreadMessageCreated):
#                 ensure_slot("text")

#             elif isinstance(event, ThreadMessageDelta):
#                 blk = event.data.delta.content[0]
#                 if isinstance(blk, TextDeltaBlock):
#                     ensure_slot("text")
#                     assistant_items[-1]["content"] += blk.text.value
#                     text_pl.markdown(assistant_items[-1]["content"], unsafe_allow_html=True)

#         if "final_report" not in st.session_state:
#             st.session_state["final_report"] = []
        
#         st.session_state["final_report"].append(assistant_items)
#         st.session_state.report_generated = True

#         st.rerun()



#         ###


#     # Display the generated report
#     if st.session_state.report_generated:
#         st.markdown(st.session_state.final_report[0][0]['content'], unsafe_allow_html=True)

#         # # Offer download as Markdown
#         # st.download_button(
#         #     "⬇️ Download report (Markdown)",
#         #     st.session_state.final_report,
#         #     file_name="scientific_report.md",
#         #     mime="text/markdown",
#         # )

#         # Optionally, add a next‑steps button to reset or exit
#         if st.button("🔄 Start new session"):
#             st.session_state.clear()
#             st.experimental_rerun()


if st.session_state.app_state == "report_generation":
    report_generation(client)
