from __future__ import annotations
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import pandas as pd



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


###############################################################


import ast          # safer than eval for “Python-looking” literals
import streamlit as st

# ── 1. Read the file (you could also use st.file_uploader) ────────────
# with open("/home/szefer/ai_assistant/ai_research_assistant/poc_scripts/hypotheses.txt", encoding="utf-8") as fp:
#     raw_text = fp.read()

# # ── 2. Convert the string → list[dict] ────────────────────────────────
# #     hypotheses.txt uses single quotes, so json.loads() would choke;
# #     ast.literal_eval understands Python-style literals.
# try:
#     hypotheses_list = ast.literal_eval(raw_text)   # ← [{'title': ...}, …]
# except (ValueError, SyntaxError) as err:
#     st.error(f"Couldn’t parse hypotheses.txt: {err}")
#     st.stop()

# ── 3. Store in session_state ─────────────────────────────────────────
# #     Ensure the surrounding dict exists, then assign.
# st.session_state.setdefault("updated_hypotheses", {})
# st.session_state.updated_hypotheses["hypotheses"] = hypotheses_list

# st.success("Hypotheses loaded into session_state ✅")

# st.session_state.app_state = "plan_execution"

###############################################################


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
if 'current_exec_idx' not in st.session_state:
    st.session_state['current_exec_idx'] = 0


###########################################################

# ————————————————————————————————————————————————————————————————
# HELPER  ✨  Markdown renderer for hypotheses
# ————————————————————————————————————————————————————————————————

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

# ————————————————————————————————————————————————————————————————
# ASSISTANTS  (basic creation – adjust model/params as needed)
# ————————————————————————————————————————————————————————————————

data_summary_assistant = client.beta.assistants.create(
    name="Data Summarizing Assistant",
    instructions=data_summary_instructions,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
    temperature=0,
)

# Additional assistants (analysis, execution) … create as needed

# ────────────────────────────────────────────────────────────────────────────────
# 🧮  SESSION STATE INIT
# ────────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = dict(
        app_state="upload",                # upload ▸ processing ▸ hypotheses_manager ▸ …
        data_uploaded=False,
        hypotheses_uploaded=False,
        processing_done=False,
        files={},                           # {filename: file‑like‑object}
        file_ids=[],
        hypotheses="",
        data_summary="",
        updated_hypotheses={},
        thread_id="",
        selected_hypothesis=0,
        analysis_plan_chat_history=[],
        plan_execution_chat_history=[]            # index of hypothesis being edited
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()








# ────────────────────────────────────────────────────────────────────────────────
# 🖋️  HELPER: MARKDOWN RENDERER FOR HYPOTHESES
# ────────────────────────────────────────────────────────────────────────────────

def render_hypothesis_md(hyp: dict) -> str:
    """Return a markdown block for a single refined hypothesis."""
    md = [f"### {hyp['title']}"]

    if hyp.get("steps"):
        md.append("**Rationale:**")
        for i, s in enumerate(hyp["steps"], start=1):
            md.append(f"{i}. {s['step']}")

    if hyp.get("final_hypothesis"):
        md.append("\n> **Final refined hypothesis**:\n>")
        md.append(f"> {hyp['final_hypothesis']}")

    return "\n".join(md)


def format_initial_assistant_msg(title: str, steps: list[dict]) -> str:
    """Return formatted markdown for the assistant seed message."""
    lines = [f"**Refined hypothesis:** {title}", "", "**Rationale:**"]
    lines += [f"{idx}. {step['step']}" for idx, step in enumerate(steps, start=1)]
    return "\n".join(lines)






# ────────────────────────────────────────────────────────────────────────────────
# 📂  SIDEBAR – UPLOADS (Stage 1 + Stage 0)
# ────────────────────────────────────────────────────────────────────────────────

if st.session_state.app_state in {"upload", "processing"}:
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
# MAIN AREA – STAGE 1  (PROCESSING)
# ────────────────────────────────────────────────────────────────────────────────

st.title("🔬 Hypotheses Workflow")

if st.session_state.app_state == "processing":
    st.subheader("Step 1 – Process files")

    # PROCESS FILES BUTTON
    if st.button("🚀 Process Files", disabled=st.session_state.processing_done):
        # 1️⃣ create thread
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

        # 2️⃣ upload files to assistant
        for _name, f in st.session_state.files.items():
            openai_file = client.files.create(file=f, purpose="assistants")
            st.session_state.file_ids.append(openai_file.id)

        # 3️⃣ update thread with resources
        client.beta.threads.update(
            thread_id=st.session_state.thread_id,
            tool_resources={"code_interpreter": {"file_ids": st.session_state.file_ids}},
        )

        # 4️⃣ run data summariser
        with st.spinner("Summarising data …"):
            run = client.beta.threads.runs.create_and_poll(
                thread_id=st.session_state.thread_id,
                assistant_id=data_summary_assistant.id,
                instructions=data_summary_instructions,
                temperature=0,
            )

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
            st.session_state.data_summary = " ".join(
                blk.text.value
                for msg in messages
                for blk in msg.content
                if blk.type == "text"
            )
            with st.expander("📊 Data summary"):
                st.markdown(f"```\n{st.session_state.data_summary}\n```) ")

        # 5️⃣ refine hypotheses
        refine_prompt = (
            f"Data summary: {st.session_state.data_summary}\n\n"
            f"Hypotheses: {st.session_state.hypotheses}\n\n"
            f"{processing_files_instruction}\n"
            "Extract individual hypotheses and refine them one by one."
        )

        response = client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": refine_prompt}],
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
                                        "title": {"type": "string"},
                                        "steps": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {"step": {"type": "string"}},
                                                "required": ["step"],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["title", "steps"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["hypotheses"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )

        st.session_state.updated_hypotheses = json.loads(response.output_text)

        # augment with extra fields & pretty initial assistant message
        for hyp in st.session_state.updated_hypotheses["hypotheses"]:
            pretty_msg = format_initial_assistant_msg(hyp["title"], hyp["steps"])
            hyp.update(
                chat_history=[{"role": "assistant", "content": pretty_msg}],
                final_hypothesis="",
            )

        st.session_state.processing_done = True
        st.success("Processing complete!", icon="✅")

    # Show refined hypotheses summary (for preview)
    if st.session_state.processing_done:
        st.subheader("Refined hypotheses")
        for hyp in st.session_state.updated_hypotheses.get("hypotheses", []):
            with st.expander(hyp["title"], expanded=False):
                st.markdown(render_hypothesis_md(hyp))

    # MOVE TO NEXT STEP BUTTON
    if st.button(
        "➡️ Move to next step",
        disabled=not st.session_state.processing_done,
        key="next_step",
    ):
        st.session_state.app_state = "hypotheses_manager"
        st.rerun()







# ────────────────────────────────────────────────────────────────────────────────
# MAIN AREA – STAGE 2  (HYPOTHESIS MANAGER)
# ────────────────────────────────────────────────────────────────────────────────

elif st.session_state.app_state == "hypotheses_manager":
    # ── SIDEBAR: list of hypotheses --------------------------------------------
    with st.sidebar:
        st.header("📑 Hypotheses")
        for idx, hyp in enumerate(st.session_state.updated_hypotheses["hypotheses"]):
            with st.expander(hyp["title"], expanded=False):
                # show either final hypothesis or rationale steps
                if hyp["final_hypothesis"]:
                    st.markdown(f"> {hyp['final_hypothesis']}")
                else:
                    for j, step in enumerate(hyp["steps"], start=1):
                        st.markdown(f"{j}. {step['step']}")

                if st.button("✏️ Edit", key=f"select_{idx}"):
                    st.session_state.selected_hypothesis = idx
                    st.rerun()

    # ── MAIN CANVAS: chat & accept button -------------------------------------
    sel_idx = st.session_state.selected_hypothesis
    sel_hyp = st.session_state.updated_hypotheses["hypotheses"][sel_idx]

    st.subheader(f"🗣️ Discussion – {sel_hyp['title']}")

    # display chat history
    for msg in sel_hyp["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat input
    user_prompt = st.chat_input("Refine this hypothesis further …", key=f"chat_input_{sel_idx}")

    if user_prompt:
        sel_hyp["chat_history"].append({"role": "user", "content": user_prompt})

        with st.spinner("Thinking …"):
            response = client.responses.create(
                model="gpt-4o",
                instructions=refinig_instructions,
                input=sel_hyp["chat_history"],
                tools=[{"type": "web_search_preview"}],
                store=False,
            )

        sel_hyp["chat_history"].append({"role": "assistant", "content": response.output_text})
        st.rerun()

    # ACCEPT BUTTON
    acc_disabled = bool(sel_hyp["final_hypothesis"])
    if st.button("✅ Accept refined hypothesis", disabled=acc_disabled, key="accept"):
        sel_hyp["final_hypothesis"] = sel_hyp["chat_history"][-1]["content"]
        st.success("Hypothesis accepted!")
        st.rerun()

    # ── AUTO‑ADVANCE -----------------------------------------------------------
    if all(h["final_hypothesis"] for h in st.session_state.updated_hypotheses["hypotheses"]):
        st.session_state.app_state = "plan_manager"  # next stage placeholder
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# STAGE‑3  ▸  ANALYSIS PLAN MANAGER
# ────────────────────────────────────────────────────────────────────────────────

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

# ----------------------------------------------------------------------------
# PLAN MANAGER STAGE
# ----------------------------------------------------------------------------

def plan_manager(client: OpenAI):
    """Stage 3 – manage analysis‑plan creation & approval."""
    if st.session_state.app_state != "plan_manager":
        return

    # ------------------------------------------------------------------
    # Sidebar: list accepted hypotheses
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Accepted hypotheses")
        for idx, h in enumerate(st.session_state.updated_hypotheses["hypotheses"]):
            title = f"Hypothesis {idx+1}"
            with st.expander(title, expanded=False):
                st.markdown(h["final_hypothesis"], unsafe_allow_html=True)
                if st.button("✏️ Work on this", key=f"select_hypo_{idx}"):
                    st.session_state["current_hypothesis_idx"] = idx
                    st.rerun()

    # Which hypothesis is in focus?
    current = st.session_state.get("current_hypothesis_idx", 0)
    hypo_obj = ensure_plan_keys(
        st.session_state.updated_hypotheses["hypotheses"][current]
    )

    st.subheader(f"Analysis Plan Manager – Hypothesis {current+1}")
    st.markdown(hypo_obj["final_hypothesis"], unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Plan generation / chat
    # ------------------------------------------------------------------
    chat_hist = hypo_obj["analysis_plan_chat_history"]

    if not chat_hist:  # first‑time plan generation
        if st.button("Generate plan", key="generate_plan"):
            prompt = (
                f"Here is the data summary: {st.session_state.data_summary}\n\n"
                f"Here is the hypothesis: {hypo_obj['final_hypothesis']}"
            )
            with st.spinner("Generating …"):
                resp = client.responses.create(
                    model="gpt-4o",
                    temperature=0,
                    instructions=analyses_step_generation_instructions,
                    input=prompt,
                    stream=False,
                    text={"format": {"type": "text"}},
                    store=False,
                )
            chat_hist.append({"role": "assistant", "content": resp.output_text})
            st.rerun()

    if not hypo_obj["analysis_plan_accepted"]:
        # Show existing chat
        for m in chat_hist:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)

        user_msg = st.chat_input("Refine this analysis plan …")
        if user_msg:
            chat_hist.append({"role": "user", "content": user_msg})
            with st.spinner("Thinking …"):
                resp = client.responses.create(
                    model="gpt-4o",
                    temperature=0,
                    instructions=analyses_step_generation_instructions,
                    input=chat_hist,
                    stream=False,
                    text={"format": {"type": "text"}},
                    store=False,
                )
            chat_hist.append({"role": "assistant", "content": resp.output_text})
            st.rerun()

        if chat_hist:
            if st.button("✅ Accept this plan", key="accept_plan"):
                hypo_obj["analysis_plan"] = chat_hist[-1]["content"]
                hypo_obj["analysis_plan_accepted"] = True
                st.rerun()

    # ------------------------------------------------------------------
    # Show accepted plan
    # ------------------------------------------------------------------
    if hypo_obj["analysis_plan_accepted"]:
        st.success("Plan accepted")
        st.markdown(hypo_obj["analysis_plan"], unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Advance once **all** hypotheses have an accepted plan
    # ------------------------------------------------------------------
    all_ready = all(
        h.get("analysis_plan") and h.get("analysis_plan_accepted")
        for h in st.session_state.updated_hypotheses["hypotheses"]
    )

    if all_ready:

        print(st.session_state.updated_hypotheses["hypotheses"])

        if st.button("➡️ Move to plan execution stage"):
            st.session_state.app_state = "plan_execution"
            st.rerun()

if st.session_state.app_state == "plan_manager":
    plan_manager(client)







###############################################################################


###############################################################################



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

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
# … imports & other stage code remain unchanged …

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
# -----------------------------------------------------------------------------
# PLAN EXECUTION STAGE
# -----------------------------------------------------------------------------

def plan_execution(client: OpenAI):
    if st.session_state.app_state != "plan_execution":
        return

    st.title("Analysis Plan Execution")

    # ------------------------------------------------------------------
    # Sidebar – hypotheses
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Accepted hypotheses")
        for idx, h in enumerate(st.session_state.updated_hypotheses["hypotheses"]):
            title = f"Hypothesis {idx+1}"
            with st.expander(title, expanded=False):
                st.markdown(h["final_hypothesis"], unsafe_allow_html=True)
                if st.button("▶️ Run / review", key=f"select_exec_{idx}"):
                    st.session_state["current_exec_idx"] = idx
                    st.rerun()

    # Which hypothesis are we executing?
    current = st.session_state.get("current_exec_idx", 0)
    hypo_obj = ensure_execution_keys(
        st.session_state.updated_hypotheses["hypotheses"][current])

    # ------------------------------------------------------------------
    # Parse analysis plan robustly
    # ------------------------------------------------------------------
    plan_dict = safe_load_plan(hypo_obj["analysis_plan"])
    if not plan_dict:
        st.error("❌ Could not parse analysis plan JSON. Please regenerate the plan in the previous stage or ask the assistant to output valid JSON.")
        return

    plan_title = plan_dict["analyses"][0]["title"]
    plan_steps = plan_dict["analyses"][0]["steps"]

    # Normalise stored plan to dict for future safety
    if isinstance(hypo_obj["analysis_plan"], str):
        hypo_obj["analysis_plan"] = plan_dict

    # ------------------------------------------------------------------
    # Main canvas – plan outline + chat / execution UI
    # ------------------------------------------------------------------
    st.subheader(f"Hypothesis {current+1}: {plan_title}")
    st.markdown("### Plan steps")
    for num, s in enumerate(plan_steps, start=1):
        st.markdown(f"{num}. {s['step']}")

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
        instructions = user_prompt if user_prompt else step_execution_instructions

        if user_prompt:
            hypo_obj["plan_execution_chat_history"].append(
                {"role": "user", "content": user_prompt}
            )

        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=f"\n\nThe analysis plan to execute:\n{json.dumps(plan_dict, indent=2)}",
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

        stream = client.beta.threads.runs.create(
            thread_id    = st.session_state.thread_id,
            assistant_id = analysis_assistant.id,
            instructions = instructions,
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

        hypo_obj["plan_execution_chat_history"].append(
            {"role": "assistant", "items": assistant_items}
        )

        st.rerun()

# -----------------------------------------------------------------------------
# ROUTER – call the appropriate stage function each run
# -----------------------------------------------------------------------------

if st.session_state.app_state == "plan_execution":
    plan_execution(client)
    