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
- You are an **expert in ecological research and statistical analysis**, with proficiency in **R**.
- You must apply the **highest quality statistical methods and approaches** in data analysis.
- Your suggestions should be based on **best practices in ecological data analysis**.
- Your role is to **provide guidance, suggestions, and recommendations** within your area of expertise.
- Students will seek your assistance with **data analysis, interpretation, and statistical methods**.
- Since students have **limited statistical knowledge**, your responses should be **simple and precise**.
- Students also have **limited programming experience**, so provide **clear and detailed instructions**.
"""


data_summary_assistant = client.beta.assistants.create(
                    name="Data Summarizing Assistant",
                    temperature=0,
                    instructions=data_summary_instructions,
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

# Uncomment this
# if "hypotheses_refined" not in st.session_state:
#     st.session_state.hypotheses_refined = True
# if "refined_hypotheses" not in st.session_state:
#     st.session_state.refined_hypotheses = {"hypotheses": [{"title": "Higher bee diversity and abundance in rural parks", "steps": [{"step": "Current Hypothesis: Wild bee diversity and abundance will be higher in rural areas compared to urban landscapes."}, {"step": "Data Support: The dataset includes 'Species code', 'Floral richness', 'Landscape type', and 'Population density' which can be used to compare rural vs. urban sites."}, {"step": "Issues: The dataset lacks direct measures of bee abundance. Observations may not be exhaustive."}, {"step": "Refined Hypothesis: Bee species richness ('Species.code') and floral richness will be higher in sites classified as rural ('Landscape.type'). 'Floral richness' should positively correlate with 'Species.code' in rural areas."}], "history": [], "final_hypotheses": []}, {"title": "Functional traits filtering by urbanization", "steps": [{"step": "Current Hypothesis: Bee communities in highly urbanized areas will be dominated by smaller, generalist bees."}, {"step": "Data Support: The dataset contains 'Mean.body.size', 'Nesting place', and 'Floral specificity' which can reflect generalist vs. specialist traits."}, {"step": "Issues: Lack of detailed trait data for specific bees to make comprehensive conclusions about all traits."}, {"step": "Refined Hypothesis: Sites with higher 'Impervious.surface.area' and 'Population.density' will have a higher proportion of small-bodied bees ('Mean.body.size') and generalists (e.g., 'Floral specificity')."}], "history": [], "final_hypotheses": []}, {"title": "Negative response of bee diversity and abundance to urbanization", "steps": [{"step": "Current Hypothesis: Bee abundance and diversity will negatively correlate with urbanization."}, {"step": "Data Support: 'Area size', 'Impervious surface area', and 'Population density' can help model urbanization, while 'Species.code' can reflect diversity."}, {"step": "Issues: Need more abundance measures. 'Species.code' provides richness, not abundance."}, {"step": "Refined Hypothesis: Richness of bee species ('Species.code') will inversely correlate with 'Impervious.surface.area' and 'Population.density'. Sites with more floral resources will have higher richness."}], "history": [], "final_hypotheses": []}, {"title": "Sex-specific responses to urbanization", "steps": [{"step": "Current Hypothesis: Female and male bees will exhibit different responses to urbanization."}, {"step": "Data Support: The 'Sex' column can reveal differences between sexes in urban vs rural contexts."}, {"step": "Issues: Dataset does not provide abundance data needed for sex-specific analyses."}, {"step": "Refined Hypothesis: Urban sites ('Population.density') will exhibit a lower female-to-male ratio ('Sex') compared to rural sites due to resource availability and nesting conditions."}], "history": [], "final_hypotheses": []}, {"title": "Beta diversity driven by turnover or nestedness", "steps": [{"step": "Current Hypothesis: \u03b2-diversity between urban and rural parks results from species turnover."}, {"step": "Data Support: The dataset provides 'Species.code' and site identifiers which can be used to calculate \u03b2-diversity."}, {"step": "Issues: Lack of explicit abundance data limits quantitative nestedness analysis."}, {"step": "Refined Hypothesis: \u03b2-diversity calculated based on 'Species.code' between rural and urban ('Landscape.type') will show more species turnover due to habitat differences."}], "history": [], "final_hypotheses": []}, {"title": "Relative contributions to gamma diversity", "steps": [{"step": "Current Hypothesis: \u03b1-diversity and \u03b2-diversity will vary, influencing \u03b3-diversity."}, {"step": "Data Support: 'Species.code' can provide insights into \u03b1-diversity and comparisons between sites for \u03b2-diversity."}, {"step": "Issues: Lack of abundance and detailed species-level richness data limits accurate \u03b3-diversity estimates."}, {"step": "Refined Hypothesis: Combine within-site species diversity ('Species.code') with between-site diversity measures to assess regional diversity influenced by landscape differences ('Landscape.type')."}], "history": [], "final_hypotheses": []}, {"title": "Trait\u2013environment interactions", "steps": [{"step": "Current Hypothesis: Specific ecological traits will show associations with environmental variables."}, {"step": "Data Support: Traits like 'Nesting place', 'Mean.body.size', and 'Social behavior' can be evaluated against 'Impervious.surface.area' and 'Landscape.diversity'."}, {"step": "Issues: Specific traits data may be too broad for nuanced analyses without additional context."}, {"step": "Refined Hypothesis: Cavity-nesting bees ('Nesting place') will associate more with urban environments ('Impervious.surface.area'), while ground-nesting bees will prevail in rural areas ('Grasslands')."}], "history": [], "final_hypotheses": []}]}
if "approved_hypotheses" not in st.session_state:
    st.session_state.approved_hypotheses = []
if 'analysis_outputs' not in st.session_state:
    st.session_state['analysis_outputs'] = {}

# UI
# Force a font accross the app
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Open+Sans&display=swap" rel="stylesheet">
<style>
* {
    font-family: 'Open Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# Header with soulless (monospace) font
st.markdown("""
<h1 style='text-align: center; font-family: monospace; letter-spacing: 2px;'>
RESEARCH ASSISTANT
</h1>
""", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns(2)


############################################################## SKIP THE STEP
# if not st.session_state['refined_hypotheses']:
#     import ast
#     with open('/home/piotr/projects/ai_research_assistant/misc/refined_hypotheses_with_key.txt', 'r') as file:
#         file_content = file.read()
#         # Convert the string to a Python object (a list, in this case)
#     data_list = ast.literal_eval(file_content)
#     # print(data_list)
#     st.session_state['refined_hypotheses']['hypotheses'] = data_list
############################################################################


if st.session_state.hypotheses_refined:
    for hypo in st.session_state['refined_hypotheses']['hypotheses']:
        hypo.setdefault('final_hypothesis_history', [])

    # print(f"NO of APPROVED HYPOTHESES: {st.session_state['approved_hypotheses']}")
    # print(f"NO of REFINED HYPOTHESES: {st.session_state['refined_hypotheses']['hypotheses']}")


    # Check if approved hypotheses have the same length as refined hypotheses
    import pprint
    # pprint.pprint(f"AH: \n\n{st.session_state.approved_analyses_plans}\n\n")
    is_every_plan_approved = all([len(hypo["analysis_plan"]) > 0 for hypo in st.session_state.approved_analyses_plans])
    print(f"\n\n[INFO] Is every plan approved? {is_every_plan_approved}\n\n")
    if is_every_plan_approved:
        
        st.title("Analysis Manager")

        # Create an assistant to execute the analysis
        analysis_assistant = client.beta.assistants.create(
            name="Analysis Assistant",
            temperature=0,
            instructions="""
            You are an expert in ecological research and statistical analysis in Python. 
            Your task is to execute the analysis plan provided by the user.
            """,
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o"
        )

        # Create a thread
        thread = client.beta.threads.create()
        st.session_state.analysis_execution_thread_id = thread.id

        analysis_container = st.container()

        with st.sidebar:
            
            st.markdown(
                """
                <span style='font-size:16px; font-weight:600;'>📄 Approved hypotheses</span>
                """, unsafe_allow_html=True)
            
            selected_hypothesis = st.selectbox(
                "Select hypothesis to run the analysis", 
                options=[hypothesis["hypothesis_title"] for hypothesis in st.session_state.approved_analyses_plans]
                )

            print(f"\n[INFO] THE SELECTED HYPOTHESIS: {selected_hypothesis}")

            analyses_steps = next(
                (item for item in st.session_state['approved_hypotheses'] if item["hypothesis_title"] == selected_hypothesis), None
            )

            analysis_dict = json.loads(analyses_steps["analysis_plan"]['content'])

            # print(f"\n[INFO] THE ANALYSIS DICT:\n{analysis_dict}\n")

            print(f"\n\nAnalysis DICT: {analysis_dict['analyses'][0]}\n\n")
            # print(type(analysis_dict["analyses"][0]))

            # analysis = json.loads(analysis_dict["analyses"][0])

            title = analysis_dict['analyses'][0]["title"]
            steps = analysis_dict['analyses'][0]["steps"]
            st.markdown(f"**{title}**")
            
            for step in steps:
                st.markdown(f"- {step['step']}")

            if st.button("Run the analysis"):
                
                with st.spinner("Running the analysis..."):            
                    
                    # Create a message in the thread
                    message = client.beta.threads.messages.create(
                        thread_id=st.session_state.analysis_execution_thread_id,
                        role="user",
                        content=f"\n\nThe analysis plan:\n{analysis_dict['analyses'][0]}\n",
                    )

                    client.beta.threads.update(
                        thread_id=st.session_state.analysis_execution_thread_id,
                        tool_resources={"code_interpreter": {"file_ids": [file_id for file_id in st.session_state.file_ids]}}
                    )

                    print(f"\n\n[INFO] The analysis plan:\n{analysis_dict['analyses'][0]}\n")
                    
                    stream = client.beta.threads.runs.create(
                        thread_id=thread.id,
                        assistant_id=analysis_assistant.id,
                        instructions="Execute the analysis plan.",
                        tool_choice={"type": "code_interpreter"},
                        stream=True,
                    )
                    
                    executor_output = []
                    
                    for event in stream:
                        if isinstance(event, ThreadRunStepCreated):
                            if event.data.step_details.type == "tool_calls":
                                executor_output.append({"type": "code_input",
                                                        "content": ""})

                                code_input_expander= st.status("Writing code ⏳ ...", expanded=True)
                                code_input_block = code_input_expander.empty()

                        if isinstance(event, ThreadRunStepDelta):
                            if event.data.delta.step_details.tool_calls[0].code_interpreter is not None:
                                code_interpretor = event.data.delta.step_details.tool_calls[0].code_interpreter
                                code_input_delta = code_interpretor.input
                                if (code_input_delta is not None) and (code_input_delta != ""):
                                    executor_output[-1]["content"] += code_input_delta
                                    code_input_block.empty()
                                    code_input_block.code(executor_output[-1]["content"])

                        elif isinstance(event, ThreadRunStepCompleted):
                            if isinstance(event.data.step_details, ToolCallsStepDetails):
                                code_interpretor = event.data.step_details.tool_calls[0].code_interpreter
                                if code_interpretor.outputs:
                                    print("***"*10)
                                    print(code_interpretor)
                                    print("***"*10)
                                    code_interpretor_outputs = code_interpretor.outputs[0]
                                    code_input_expander.update(label="Code", state="complete", expanded=False)
                                    # Image
                                    if isinstance(code_interpretor_outputs, CodeInterpreterOutputImage):
                                        image_html_list = []
                                        for output in code_interpretor.outputs:
                                            image_file_id = output.image.file_id
                                            image_data = client.files.content(image_file_id)
                                            
                                            # Save file
                                            image_data_bytes = image_data.read()
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

                                        executor_output.append({"type": "image",
                                                                "content": image_html_list})
                                    # Console log
                                    elif isinstance(code_interpretor_outputs, CodeInterpreterOutputLogs):
                                        executor_output.append({"type": "code_output",
                                                                "content": ""})
                                        code_output = code_interpretor.outputs[0].logs
                                        with st.status("Results", state="complete"):
                                            st.code(code_output)    
                                            executor_output[-1]["content"] = code_output   

                        elif isinstance(event, ThreadMessageCreated):
                            executor_output.append({"type": "text",
                                                    "content": ""})
                            assistant_text_box = st.empty()

                        elif isinstance(event, ThreadMessageDelta):
                            if isinstance(event.data.delta.content[0], TextDeltaBlock):
                                assistant_text_box.empty()
                                executor_output[-1]["content"] += event.data.delta.content[0].text.value
                                assistant_text_box.markdown(executor_output[-1]["content"])

        st.stop()

    # Check that each hypothesis has non-empty final_hypothesis
    if all(len(hypo['final_hypothesis']) > 0 for hypo in st.session_state['refined_hypotheses']['hypotheses']):
        final_hypotheses_list = [
            hypo['final_hypothesis']['content']
            for hypo in st.session_state['refined_hypotheses']['hypotheses']
        ]

        st.subheader("Analysis Plan Manager")

        # Loop over each hypothesis
        for i, hypothesis in enumerate(final_hypotheses_list):
            
            with st.expander(f"Hypothesis {i+1}"):

                # Display the refined hypothesis text
                st.markdown(hypothesis)

                # Get the plan message history for this hypothesis (if any)
                message_history = st.session_state.refined_hypotheses['hypotheses'][i]['final_hypothesis_history']

                # If there's already at least one plan message, we consider it "generated"
                plan_already_generated = len(message_history) > 0

                if plan_already_generated:
                    # (A) Plan is already generated => Show the plan details
                    for message in message_history:
                        with st.chat_message(message['role']):
                            plan = json.loads(message['content'])
                            st.markdown(f"**{plan['analyses'][0]['title']}**")
                            for step in plan["analyses"][0]["steps"]:
                                st.markdown(f"- {step['step']}")

                    # Show the Accept button if a plan is generated
                    accept_button = st.button("Accept the plan", key=f"accept_plan_{i}")
                    if accept_button:
                        # On accept, store the last plan as final_hypothesis
                        st.session_state.approved_hypotheses.append({"hypothesis_title": hypothesis, "analysis_plan": message_history[-1]})
                        st.session_state.approved_analyses_plans.append({"hypothesis_title": hypothesis, "analysis_plan": message_history[-1]})
                        st.session_state.refined_hypotheses['hypotheses'][i]['final_hypothesis'] = message_history[-1]

                        st.rerun()

                else:
                    # (B) No plan generated => Show "Generate plan" button
                    generate_button = st.button(f"Generate a plan to test the Hypothesis {i+1}.", key=f"generate_plan_{i}")
                    if generate_button:
                        plan_steps_generation_input = f"""
                        Here is the data summary: {st.session_state.data_summary}\n
                        Here is the hypothesis: {hypothesis}.
                        """
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
                        st.session_state.refined_hypotheses['hypotheses'][i]['final_hypothesis_history'].append(new_plan)

                        # Rerun the script so that next time we detect the plan is generated
                        st.rerun()

        st.stop()

        if st.button("Save the refined hypotheses and prepare analysis plan"):
            # print(st.session_state['refined_hypotheses']['hypotheses'])
            print(f"\n This should save the hypotheses to the approved_hpotheses:\n")
            print(f"\n\n[INFO] The final hypotheses:\n{st.session_state['refined_hypotheses']['hypotheses']}\n")

    else:
        print("Some hypotheses have an empty 'final_hypothesis'.")


if st.session_state.hypotheses_refined:
    st.title("Hypothesis Manager")

    # print(f"\n\nUPDATED HYPOTHESES WITH HISTORY:\n{st.session_state.refined_hypotheses}\n\n")

    updated_hypotheses = st.session_state.refined_hypotheses

    for i, hypothesis_obj in enumerate(updated_hypotheses["hypotheses"]):
        with st.expander(f"{hypothesis_obj['title']}"):

            if st.session_state['refined_hypotheses']['hypotheses'][i]['final_hypothesis']:
                refined_hypothesis = st.session_state['refined_hypotheses']['hypotheses'][i]['final_hypothesis']
                st.markdown(refined_hypothesis['content'])

            # Final hypothesis is empty
            elif st.session_state['refined_hypotheses']['hypotheses'][i]['final_hypothesis'] == []:
                for step_j, step_obj in enumerate(hypothesis_obj["steps"]):
                    st.markdown(f"**Step {step_j+1}:** {step_obj['step']}")
                
                st.markdown("---")

                # Display history of the previous conversations about this particular hypothesis.
                for msg in st.session_state['refined_hypotheses']['hypotheses'][i]['history']:
                    # print(f"\n\nMESSAGE: {msg}")
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                prompt = st.chat_input(
                    "Discuss with the assistant to refine this hypothesis further.",
                    key=f"chat_input{i}")
                
                button = st.button("Accept the refined hypotheses", key=f"button{i}")
                
                if button:
                    # Saves the last message from the history.
                    last_message = st.session_state['refined_hypotheses']['hypotheses'][i]['history'][-1]
                    print(f"\n\n[INFO] The last message in the refined hypotheses history:\n{last_message}\n")
                    # Here the last hypotheses should be saved as accepted_hypotheses
                    st.session_state['approved_hypotheses'].append(
                        {"hypothesis_title": hypothesis_obj['title'],
                         "content": last_message})
                    st.session_state['refined_hypotheses']['hypotheses'][i]['final_hypothesis'] = last_message
                    st.rerun()


                if prompt:
                    st.session_state['refined_hypotheses']['hypotheses'][i]['history'].append(
                        {"role":"user", "content": prompt}
                    )
                    history = st.session_state['refined_hypotheses']['hypotheses'][i]['history']
                    response = client.responses.create(
                        model="gpt-4o",
                        instructions=refinig_instructions,
                        input=history,
                        tools=[{"type": "web_search_preview"}],
                        store=False
                    )
                    # st.write(response.output_text)
                    st.session_state['refined_hypotheses']['hypotheses'][i]['history'].append(
                    {"role":"assistant", "content": response.output_text}
                    )
                    st.rerun()

                # pass

# CSV upload on the left
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

# TXT upload on the right
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


# Three columns to center the button in the middle one
if st.button("🚀 Process Files"):
    if st.session_state.data_uploaded and st.session_state.hypotheses_uploaded:
        
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

                # print(f"[INFO] The RAW list response attributes :\n\n{dir(messages)}\n\n")

                messages_list = list(messages)
                
                assistant_response = []
                
                for msg in messages_list:
                    for block in msg.content:
                        if block.type == 'text':
                            # print(block.text.value)
                            # st.write(block.text.value)
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
    
            for i, hypothesis_obj in enumerate(updated_hypotheses["hypotheses"]):
                # print(f"HYPOTHESIS OBJECT: {hypothesis_obj}")
                # Add a conversation history to each element.
                # print(f"HYPOTHESIS OBJ TITLE: {hypothesis_obj['title']}")
                # print(f"HYPOTHESIS OBJ STEPS: {hypothesis_obj['steps']}")

                combined_str = hypothesis_obj['title'] + "\n\nHYPOTHESIS STEPS:\n"
                for s in hypothesis_obj['steps']:
                    combined_str += f"- {s['step']}\n"

                updated_hypotheses['hypotheses'][i]['history'] = [
                    {"role": "user", "content": combined_str}
                    ]
                
                # Add a refined hypotheses to the dict.
                updated_hypotheses['hypotheses'][i]['final_hypothesis'] = []
                updated_hypotheses['hypotheses'][i]['final_hypothesis_history'] = []

            st.session_state.refined_hypotheses = updated_hypotheses
            st.session_state.hypotheses_refined = True

            # Success
            st.success("Done")
            st.rerun()
    else:
        st.warning("Upload your data and hypotheses first!")