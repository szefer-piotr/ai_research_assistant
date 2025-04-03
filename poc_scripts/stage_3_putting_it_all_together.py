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
# Uncomment this
# if "hypotheses_refined" not in st.session_state:
#     st.session_state.hypotheses_refined = True
# if "refined_hypotheses" not in st.session_state:
#     st.session_state.refined_hypotheses = {"hypotheses": [{"title": "Higher bee diversity and abundance in rural parks", "steps": [{"step": "Current Hypothesis: Wild bee diversity and abundance will be higher in rural areas compared to urban landscapes."}, {"step": "Data Support: The dataset includes 'Species code', 'Floral richness', 'Landscape type', and 'Population density' which can be used to compare rural vs. urban sites."}, {"step": "Issues: The dataset lacks direct measures of bee abundance. Observations may not be exhaustive."}, {"step": "Refined Hypothesis: Bee species richness ('Species.code') and floral richness will be higher in sites classified as rural ('Landscape.type'). 'Floral richness' should positively correlate with 'Species.code' in rural areas."}], "history": [], "final_hypotheses": []}, {"title": "Functional traits filtering by urbanization", "steps": [{"step": "Current Hypothesis: Bee communities in highly urbanized areas will be dominated by smaller, generalist bees."}, {"step": "Data Support: The dataset contains 'Mean.body.size', 'Nesting place', and 'Floral specificity' which can reflect generalist vs. specialist traits."}, {"step": "Issues: Lack of detailed trait data for specific bees to make comprehensive conclusions about all traits."}, {"step": "Refined Hypothesis: Sites with higher 'Impervious.surface.area' and 'Population.density' will have a higher proportion of small-bodied bees ('Mean.body.size') and generalists (e.g., 'Floral specificity')."}], "history": [], "final_hypotheses": []}, {"title": "Negative response of bee diversity and abundance to urbanization", "steps": [{"step": "Current Hypothesis: Bee abundance and diversity will negatively correlate with urbanization."}, {"step": "Data Support: 'Area size', 'Impervious surface area', and 'Population density' can help model urbanization, while 'Species.code' can reflect diversity."}, {"step": "Issues: Need more abundance measures. 'Species.code' provides richness, not abundance."}, {"step": "Refined Hypothesis: Richness of bee species ('Species.code') will inversely correlate with 'Impervious.surface.area' and 'Population.density'. Sites with more floral resources will have higher richness."}], "history": [], "final_hypotheses": []}, {"title": "Sex-specific responses to urbanization", "steps": [{"step": "Current Hypothesis: Female and male bees will exhibit different responses to urbanization."}, {"step": "Data Support: The 'Sex' column can reveal differences between sexes in urban vs rural contexts."}, {"step": "Issues: Dataset does not provide abundance data needed for sex-specific analyses."}, {"step": "Refined Hypothesis: Urban sites ('Population.density') will exhibit a lower female-to-male ratio ('Sex') compared to rural sites due to resource availability and nesting conditions."}], "history": [], "final_hypotheses": []}, {"title": "Beta diversity driven by turnover or nestedness", "steps": [{"step": "Current Hypothesis: \u03b2-diversity between urban and rural parks results from species turnover."}, {"step": "Data Support: The dataset provides 'Species.code' and site identifiers which can be used to calculate \u03b2-diversity."}, {"step": "Issues: Lack of explicit abundance data limits quantitative nestedness analysis."}, {"step": "Refined Hypothesis: \u03b2-diversity calculated based on 'Species.code' between rural and urban ('Landscape.type') will show more species turnover due to habitat differences."}], "history": [], "final_hypotheses": []}, {"title": "Relative contributions to gamma diversity", "steps": [{"step": "Current Hypothesis: \u03b1-diversity and \u03b2-diversity will vary, influencing \u03b3-diversity."}, {"step": "Data Support: 'Species.code' can provide insights into \u03b1-diversity and comparisons between sites for \u03b2-diversity."}, {"step": "Issues: Lack of abundance and detailed species-level richness data limits accurate \u03b3-diversity estimates."}, {"step": "Refined Hypothesis: Combine within-site species diversity ('Species.code') with between-site diversity measures to assess regional diversity influenced by landscape differences ('Landscape.type')."}], "history": [], "final_hypotheses": []}, {"title": "Trait\u2013environment interactions", "steps": [{"step": "Current Hypothesis: Specific ecological traits will show associations with environmental variables."}, {"step": "Data Support: Traits like 'Nesting place', 'Mean.body.size', and 'Social behavior' can be evaluated against 'Impervious.surface.area' and 'Landscape.diversity'."}, {"step": "Issues: Specific traits data may be too broad for nuanced analyses without additional context."}, {"step": "Refined Hypothesis: Cavity-nesting bees ('Nesting place') will associate more with urban environments ('Impervious.surface.area'), while ground-nesting bees will prevail in rural areas ('Grasslands')."}], "history": [], "final_hypotheses": []}]}
if "approved_hypotheses" not in st.session_state:
    st.session_state.approved_hypotheses = {}


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
import ast
with open('/home/piotr/projects/ai_research_assistant/misc/refined_hypotheses.txt', 'r') as file:
    file_content = file.read()
# Convert the string to a Python object (a list, in this case)
data_list = ast.literal_eval(file_content)
print(data_list)
st.session_state['refined_hypotheses']['hypotheses'] = data_list
############################################################################


# See if all refined hypotheses have their final versions approved
if st.session_state['refined_hypotheses']:
    if all(len(hypo['final_hypothesis']) > 0 for hypo in st.session_state['refined_hypotheses']['hypotheses']):
        final_hypotheses_list = [hypo['final_hypothesis']['content'] for hypo in st.session_state['refined_hypotheses']['hypotheses']]
        st.subheader("Analysis Plan Manager")  
        for i, hypothesis in enumerate(final_hypotheses_list):
            with st.expander(f"Hypothesis {i+1}"):
                st.markdown(hypothesis)
                button = st.button(f"Generate a plan to test the Hypothesis {i+1}.", key=hypothesis)
                if button:
                    st.write(f"Here is your plan: a, b, c, do it yourself {hypothesis}:P")
        
        st.stop()

        if st.button("Save the refined hypotheses and prepare analysis plan"):
            print(st.session_state['refined_hypotheses']['hypotheses'])
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

            elif st.session_state['refined_hypotheses']['hypotheses'][i]['final_hypothesis'] == []:
                for step_j, step_obj in enumerate(hypothesis_obj["steps"]):
                    st.markdown(f"**Step {step_j+1}:** {step_obj['step']}")
                
                st.markdown("---")

                # Display history of the previous conversations about this particular hypothesis.
                for msg in st.session_state['refined_hypotheses']['hypotheses'][i]['history']:
                    # print(f"\n\nMESSAGE: {msg}")
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                prompt = st.chat_input(
                    "Discuss with the assistant to refine this hypothesis further.",
                    key=f"chat_input{i}")
                
                button = st.button("Accept the refined hypotheses", key=f"button{i}")
                
                if button:
                    # Saves the last message from the history.
                    last_message = st.session_state['refined_hypotheses']['hypotheses'][i]['history'][-1]
                    print(f"\n\n[INFO] The last message in the refined hypotheses history:\n{last_message}\n")
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

            st.session_state.refined_hypotheses = updated_hypotheses
            st.session_state.hypotheses_refined = True

            # Success
            st.success("Done")
            st.rerun()
    else:
        st.warning("Upload your data and hypotheses first!")