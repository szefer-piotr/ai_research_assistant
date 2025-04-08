import os
import json
import time
import base64
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

# Custom imports from your code base
from assistant_event_handlers import EventHandler

# From your custom openAI beta libraries (example placeholders)
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

################################################################################
#                               CONFIG & SETUP                                 #
################################################################################

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Instructions
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
- You are an **expert in ecological research and statistical analysis**, with proficiency in **R**.
- You must apply the **highest quality statistical methods and approaches** in data analysis.
- Your suggestions should be based on **best practices in ecological data analysis**.
- Your role is to **provide guidance, suggestions, and recommendations** within your area of expertise.
- Students will seek your assistance with **data analysis, interpretation, and statistical methods**.
- Since students have **limited statistical knowledge**, your responses should be **simple and precise**.
- Students also have **limited programming experience**, so provide **clear and detailed instructions**.
"""

# Create specialized assistants
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
    instructions="""
    You are an expert in ecological data analysis and statistical analysis in Python. 
    Your task is to write and execute every analysis step in a plan provided by the user.
    """,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
)

################################################################################
#                               SESSION STATE                                  #
################################################################################

def initialize_session_state() -> None:
    """
    Initialize session_state variables if they do not exist.
    """
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
    if "hypotheses_refined" not in st.session_state:
        st.session_state.hypotheses_refined = False
    if "refined_hypotheses" not in st.session_state:
        st.session_state.refined_hypotheses = {}
    if "approved_hypotheses" not in st.session_state:
        st.session_state.approved_hypotheses = []
    if 'analysis_outputs' not in st.session_state:
        st.session_state['analysis_outputs'] = {}

################################################################################
#                               HELPER FUNCTIONS                               #
################################################################################

def create_thread_and_upload_files() -> None:
    """
    Creates a new thread on the OpenAI Beta API, uploads each file in session state,
    and updates the thread with the file IDs.
    """
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

    for file_name, file_obj in st.session_state.files.items():
        openai_file = client.files.create(file=file_obj, purpose="assistants")
        st.session_state.file_ids.append(openai_file.id)
    
    client.beta.threads.update(
        thread_id=st.session_state.thread_id,
        tool_resources={
            "code_interpreter": {
                "file_ids": [f_id for f_id in st.session_state.file_ids]
            }
        }
    )

def run_data_summary_assistant() -> None:
    """
    Runs the Data Summarizing Assistant to generate a data summary 
    and stores the summary in session_state.data_summary.
    """
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
        # Store summary in session state
        st.session_state.data_summary = " ".join(assistant_response)

def refine_hypotheses() -> None:
    """
    Takes the data summary and raw hypotheses from session_state,
    sends them to the model for refinement, 
    and stores the refined hypotheses in session_state.refined_hypotheses.
    """
    prompt = (
        f"Data summary preview: {st.session_state.data_summary}.\n\n"
        f"Hypotheses: {st.session_state.hypotheses}.\n\n"
        f"{processing_files_instruction}\n"
        "Extract individual hypotheses from the text provided by the user and refine them one by one."
    )
    response = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": prompt}],
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
                    "required": ["hypotheses"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

    updated_hypotheses = json.loads(response.output_text)

    # Populate each hypothesis with extra fields for conversation
    for i, hypothesis_obj in enumerate(updated_hypotheses["hypotheses"]):
        combined_str = hypothesis_obj['title'] + "\n\nHYPOTHESIS STEPS:\n"
        for step in hypothesis_obj['steps']:
            combined_str += f"- {step['step']}\n"

        updated_hypotheses["hypotheses"][i]['history'] = [
            {"role": "user", "content": combined_str}
        ]
        updated_hypotheses["hypotheses"][i]['final_hypothesis'] = []
        updated_hypotheses["hypotheses"][i]['final_hypothesis_history'] = []

    st.session_state.refined_hypotheses = updated_hypotheses
    st.session_state.hypotheses_refined = True


################################################################################
#                          MAIN LOGIC & UI LAYOUT                              #
################################################################################

def display_header() -> None:
    """
    Displays the main header and sets up the page config.
    """
    st.set_page_config(
        page_title="Research assistant",
        page_icon=":tada:",
        layout="wide"
    )

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Open+Sans&display=swap" rel="stylesheet">
        <style>
        * {
            font-family: 'Open Sans', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 style='text-align: center; font-family: monospace; letter-spacing: 2px;'>
        RESEARCH ASSISTANT
        </h1>
        """,
        unsafe_allow_html=True
    )

def upload_data_files_ui() -> None:
    """
    Renders UI components for CSV and TXT uploads.
    Updates session_state with file objects and content.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<span style='font-size:16px; font-weight:600;'>📄 Upload your dataset</span>",
            unsafe_allow_html=True
        )
        csv_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
        if csv_file:
            st.toast("CSV file uploaded!")  # Remove/replace if not supported
            df = pd.read_csv(csv_file)
            st.session_state.files[csv_file.name] = csv_file
            st.session_state.data_uploaded = True
            with st.expander("Data preview."):
                st.dataframe(df.head())

    with col2:
        st.markdown(
            "<span style='font-size:16px; font-weight:600;'>📝 Upload your hypotheses</span>",
            unsafe_allow_html=True
        )
        txt_file = st.file_uploader("Choose a TXT file", type="txt", key="txt_uploader")
        if txt_file:
            st.toast("TXT file uploaded!")  # Remove/replace if not supported
            text = txt_file.read().decode("utf-8")
            st.session_state.hypotheses = text
            st.session_state.hypotheses_uploaded = True
            with st.expander("Hypotheses preview."):
                st.text_area("File content", text, height=182)

    # Process Files button
    if st.button("🚀 Process Files"):
        if st.session_state.data_uploaded and st.session_state.hypotheses_uploaded:
            create_thread_and_upload_files()
            with st.spinner("Refining your hypotheses..."):
                run_data_summary_assistant()
                st.expander("Data summary").write(st.session_state.data_summary)
                refine_hypotheses()

            st.success("Done")
            st.rerun()
        else:
            st.warning("Upload your data and hypotheses first!")

def display_hypothesis_manager() -> None:
    """
    Displays the logic for refining hypotheses and generating final analysis plans.
    Allows conversation-based refinement and acceptance of final hypotheses.
    """
    st.title("Hypothesis Manager")

    if not st.session_state.hypotheses_refined:
        st.info("No refined hypotheses to manage. Please upload files and process them first.")
        return

    updated_hypotheses = st.session_state.refined_hypotheses

    for i, hypothesis_obj in enumerate(updated_hypotheses["hypotheses"]):
        with st.expander(hypothesis_obj['title']):
            # If the final hypothesis is already set
            if hypothesis_obj['final_hypothesis']:
                st.markdown(hypothesis_obj['final_hypothesis']['content'])
                continue

            # Otherwise, show the steps and chat
            for step_j, step_obj in enumerate(hypothesis_obj["steps"]):
                st.markdown(f"**Step {step_j+1}:** {step_obj['step']}")
            st.markdown("---")

            # Display conversation history
            for msg in hypothesis_obj['history']:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            prompt = st.chat_input(
                "Discuss with the assistant to refine this hypothesis further.",
                key=f"chat_input_{i}"
            )

            # Accept button: if pressed, accept last assistant message as final
            if st.button("Accept the refined hypotheses", key=f"button_{i}"):
                if hypothesis_obj['history']:
                    # Last assistant message in the history
                    last_message = hypothesis_obj['history'][-1]
                    # Store the actual hypothesis title, not the entire text
                    st.session_state.approved_hypotheses.append({
                        "hypothesis_title": hypothesis_obj["title"],
                        "analysis_plan": last_message
                    })
                    hypothesis_obj['final_hypothesis'] = last_message
                st.rerun()

            # If user typed something, pass it to the model
            if prompt:
                hypothesis_obj['history'].append({"role": "user", "content": prompt})
                history = hypothesis_obj['history']
                response = client.responses.create(
                    model="gpt-4o",
                    instructions=refinig_instructions,
                    input=history,
                    tools=[{"type": "web_search_preview"}],
                    store=False
                )
                hypothesis_obj['history'].append({"role": "assistant", "content": response.output_text})
                st.rerun()

def display_analysis_plan_manager() -> None:
    """
    Displays the logic for final hypotheses that need an analysis plan.
    Allows generating, reviewing, and accepting an analysis plan.
    Also handles the 'Run the analysis' button and streaming outputs.
    """
    # Check if we have final hypotheses
    if not st.session_state.refined_hypotheses:
        return

    # Ensure 'final_hypothesis_history' is present on each hypothesis
    for hypo in st.session_state.refined_hypotheses["hypotheses"]:
        hypo.setdefault('final_hypothesis_history', [])

    all_hypotheses = st.session_state.refined_hypotheses["hypotheses"]

    # If user has accepted a plan for each hypothesis
    if len(st.session_state.approved_hypotheses) == len(all_hypotheses):
        st.markdown("### All Hypotheses Approved")
        st.info("You can run analyses for your hypotheses in the sidebar below.")

        if not st.session_state.analysis_execution_thread_id:
            thread = client.beta.threads.create()
            st.session_state.analysis_execution_thread_id = thread.id

        analysis_container = st.container()

        with st.sidebar:
            st.markdown(
                "<span style='font-size:16px; font-weight:600;'>📄 Approved hypotheses</span>",
                unsafe_allow_html=True
            )
            # Use the stored "hypothesis_title" for the dropdown
            options = [h["hypothesis_title"] for h in st.session_state.approved_hypotheses]
            selected_hypothesis = st.selectbox("Select hypothesis to run the analysis", options)
            st.write(f"SELECTED HYPOTHESIS: {selected_hypothesis}")

            # Grab the selected hypothesis object from approved_hypotheses
            analyses_steps = next(
                (item for item in st.session_state['approved_hypotheses'] 
                 if item["hypothesis_title"] == selected_hypothesis), 
                None
            )
            if not analyses_steps:
                st.warning("No analysis steps found.")
                return

            try:
                analysis_dict = json.loads(analyses_steps["analysis_plan"]['content'])
            except:
                st.warning("Invalid analysis plan content.")
                return

            # Display the plan
            title = analysis_dict['analyses'][0]["title"]
            steps = analysis_dict['analyses'][0]["steps"]
            st.markdown(f"**{title}**") 
            for step in steps:
                st.markdown(f"- {step['step']}")

            # Prepare a placeholder in st.session_state for storing run outputs
            if selected_hypothesis not in st.session_state['analysis_outputs']:
                st.session_state['analysis_outputs'][selected_hypothesis] = []

            # Run the analysis
            if st.button("Run the analysis"):
                with st.spinner("Running the analysis..."):
                    client.beta.threads.messages.create(
                        thread_id=st.session_state.analysis_execution_thread_id,
                        role="user",
                        content=f"\n\nThe analysis plan:\n{analysis_dict['analyses'][0]}\n"
                    )
                    client.beta.threads.update(
                        thread_id=st.session_state.analysis_execution_thread_id,
                        tool_resources={"code_interpreter": {"file_ids": st.session_state.file_ids}}
                    )

                    stream = client.beta.threads.runs.create(
                        thread_id=st.session_state.analysis_execution_thread_id,
                        assistant_id=analysis_assistant.id,
                        instructions="Execute the analysis plan on the dataset attached to the thread.",
                        tool_choice={"type": "code_interpreter"},
                        stream=True,
                    )

                # Process streamed events
                for event in stream:
                    if isinstance(event, ThreadRunStepCreated):
                        if event.data.step_details.type == "tool_calls":
                            st.session_state['analysis_outputs'][selected_hypothesis].append({
                                "type": "code_input",
                                "content": ""
                            })

                    elif isinstance(event, ThreadRunStepDelta):
                        code_interpretor = event.data.delta.step_details.tool_calls[0].code_interpreter
                        if code_interpretor is not None:
                            code_input_delta = code_interpretor.input
                            if code_input_delta:
                                st.session_state['analysis_outputs'][selected_hypothesis][-1]["content"] += code_input_delta

                    elif isinstance(event, ThreadRunStepCompleted):
                        if isinstance(event.data.step_details, ToolCallsStepDetails):
                            code_interpretor = event.data.step_details.tool_calls[0].code_interpreter
                            if code_interpretor.outputs:
                                output_obj = code_interpretor.outputs[0]
                                if isinstance(output_obj, CodeInterpreterOutputImage):
                                    image_html_list = []
                                    for o in code_interpretor.outputs:
                                        if isinstance(o, CodeInterpreterOutputImage):
                                            image_data = client.files.content(o.image.file_id).read()
                                            file_path = f"images/{o.image.file_id}.png"
                                            with open(file_path, "wb") as f:
                                                f.write(image_data)
                                            with open(file_path, "rb") as f:
                                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                                            image_html = f'<p align="center"><img src="data:image/png;base64,{b64_data}" width="600"></p>'
                                            image_html_list.append(image_html)
                                    st.session_state['analysis_outputs'][selected_hypothesis].append({
                                        "type": "image",
                                        "content": image_html_list
                                    })

                                elif isinstance(output_obj, CodeInterpreterOutputLogs):
                                    logs = output_obj.logs
                                    st.session_state['analysis_outputs'][selected_hypothesis].append({
                                        "type": "code_output",
                                        "content": logs
                                    })

                    elif isinstance(event, ThreadMessageCreated):
                        st.session_state['analysis_outputs'][selected_hypothesis].append({
                            "type": "text",
                            "content": ""
                        })

                    elif isinstance(event, ThreadMessageDelta):
                        msg = event.data.delta.content[0]
                        if hasattr(msg, "text") and msg.text.value:
                            st.session_state['analysis_outputs'][selected_hypothesis][-1]["content"] += msg.text.value

            # Show all outputs
            with analysis_container:
                st.markdown("## Analysis Output")
                outputs = st.session_state['analysis_outputs'].get(selected_hypothesis, [])
                for item in outputs:
                    if item["type"] == "text":
                        st.markdown(item["content"])
                    elif item["type"] == "code_input":
                        st.code(item["content"])
                    elif item["type"] == "code_output":
                        st.code(item["content"])
                    elif item["type"] == "image":
                        for image_html in item["content"]:
                            st.markdown(image_html, unsafe_allow_html=True)

    else:
        # If not all final hypotheses have been turned into plans
        # or the user hasn't accepted them, handle that path:
        st.subheader("Analysis Plan Manager")
        # Check if each hypothesis has a final hypothesis
        all_final = all(len(h.get('final_hypothesis', [])) > 0 for h in all_hypotheses)

        if all_final:
            for i, hypo in enumerate(all_hypotheses):
                with st.expander(f"Hypothesis {i+1}"):
                    final_hypo_text = hypo['final_hypothesis']['content']
                    st.markdown(final_hypo_text)

                    plan_history = hypo.get('final_hypothesis_history', [])
                    plan_already_generated = len(plan_history) > 0

                    if plan_already_generated:
                        for msg in plan_history:
                            with st.chat_message(msg['role']):
                                plan_obj = json.loads(msg['content'])
                                st.markdown(f"**{plan_obj['analyses'][0]['title']}**")
                                for step in plan_obj["analyses"][0]["steps"]:
                                    st.markdown(f"- {step['step']}")

                        accept_button = st.button("Accept the plan", key=f"accept_plan_{i}")
                        if accept_button:
                            # Store actual hypothesis title from `hypo`
                            st.session_state.approved_hypotheses.append({
                                "hypothesis_title": hypo['title'],
                                "analysis_plan": plan_history[-1]
                            })
                            hypo['final_hypothesis'] = plan_history[-1]
                            st.rerun()

                    else:
                        generate_button = st.button(f"Generate a plan to test Hypothesis {i+1}", key=f"generate_plan_{i}")
                        if generate_button:
                            plan_input = f"Here is the data summary: {st.session_state.data_summary}\nHere is the hypothesis: {final_hypo_text}."
                            response = client.responses.create(
                                model="gpt-4o",
                                temperature=0,
                                instructions=analyses_step_generation_instructions,
                                input=plan_input,
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
                            new_plan = {
                                "role": "assistant",
                                "content": response.output_text
                            }
                            hypo['final_hypothesis_history'].append(new_plan)
                            st.rerun()
        else:
            st.warning("Some hypotheses are missing a final refined hypothesis. Please refine them first.")

def main():
    """
    Main function to orchestrate the entire Streamlit app logic.
    """
    display_header()
    initialize_session_state()
    upload_data_files_ui()

    # If user already refined hypotheses, show the Hypothesis Manager
    display_hypothesis_manager()

    # If user has final hypotheses that need analysis planning, manage that
    display_analysis_plan_manager()

if __name__ == "__main__":
    main()
