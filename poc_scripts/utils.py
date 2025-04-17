from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import streamlit as st

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

### INSTRUCTIONS

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

code_generation_instructions = """
## Role
You are an exert in writing Python code for ecological data analysis.
## Task
Write and execute code when necessary.
Do not write code if not explicitly asked to!
When you write the code describe staps taken and provide interpretation of the results.
## Requirements
- Write clear code.
- Break the code into simple and easily executable steps.
"""

### ASSISTANTS DEFINITIONS

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


#### SCHEMAS FOR STRUCTURED OUTPUTS

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

REFINED_HYPOTHESES_EXTRACTED_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
            "title": {
                "type": "string"
            },
            "text": {
                "type": "string"
            }
            },
            "required": ["title", "text"],
            "additionalProperties": False
        }
        }
    },
    "required": ["hypotheses"],
    "additionalProperties": False
    }

REFINED_HYPOTHESES_EXTRACTED_SCHEMA_SIMPLE = {
  "type": "object",
  "properties": {
    "title": {
      "type": "string"
    },
    "text": {
      "type": "string"
    }
  },
  "required": ["title", "text"],
  "additionalProperties": False
}



def handle_stream_events(st, client, data_summary_assistant, thread_id, tools_dict):
    """
    Process the streamed events from client.beta.threads.runs.create() and update 
    the Streamlit UI (st) as well as session state accordingly.
    """
    stream = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=data_summary_assistant.id,
        temperature=0,
        stream=True
    )

    assistant_output = []
    code_input_expander = None
    code_input_block = None
    assistant_text_box = None

    for event in stream:
        # print(event)

        if isinstance(event, ThreadRunStepCreated):
            if event.data.step_details.type == "tool_calls":
                assistant_output.append({"type": "code_input", "content": ""})
                code_input_expander = st.status("Writing code ⏳ ...", expanded=True)
                code_input_block = code_input_expander.empty()

        elif isinstance(event, ThreadRunStepDelta):
            if event.data.delta.step_details.tool_calls:
                ci = event.data.delta.step_details.tool_calls[0].code_interpreter
                if ci is not None and ci.input:
                    assistant_output[-1]["content"] += ci.input
                    code_input_block.empty()
                    code_input_block.code(assistant_output[-1]["content"])

        elif isinstance(event, ThreadRunStepCompleted):
            if isinstance(event.data.step_details, ToolCallsStepDetails):
                ci = event.data.step_details.tool_calls[0].code_interpreter
                if ci.outputs and len(ci.outputs) > 0:
                    code_input_expander.update(label="Code", state="complete", expanded=False)
                    code_interpretor_outputs = ci.outputs[0]

                    # Image
                    if isinstance(code_interpretor_outputs, CodeInterpreterOutputImage):
                        image_html_list = []
                        for output in ci.outputs:
                            image_file_id = output.image.file_id
                            image_data = client.files.content(image_file_id)
                            image_data_bytes = image_data.read()

                            # Write the image to disk
                            os.makedirs("images", exist_ok=True)
                            with open(f"images/{image_file_id}.png", "wb") as file:
                                file.write(image_data_bytes)

                            # Convert to base64
                            with open(f"images/{image_file_id}.png", "rb") as f:
                                contents = f.read()
                                data_url = base64.b64encode(contents).decode("utf-8")

                            # Display in Streamlit
                            image_html = f'<p align="center"><img src="data:image/png;base64,{data_url}" width=600></p>'
                            st.html(image_html)
                            image_html_list.append(image_html)

                        assistant_output.append({"type": "image", "content": image_html_list})

                    # Console log
                    elif isinstance(code_interpretor_outputs, CodeInterpreterOutputLogs):
                        assistant_output.append({"type": "code_output", "content": ""})
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

    return assistant_output


def display_messages(messages):
    """
    Renders all messages from a list of messages. Each message is expected
    to have:
      - message["role"]: The role (e.g., "assistant", "user")
      - message["items"]: A list of dicts, each representing some content
        with a "type" (e.g., "text", "image", "code_input", "code_output")
        and "content" (the actual data to display).
    """
    for message in messages:
        with st.chat_message(message["role"]):
            for item in message["items"]:
                item_type = item["type"]
                if item_type == "text":
                    st.markdown(item["content"])
                elif item_type == "image":
                    # Each item["content"] could be a list of HTML snippets
                    for image_html in item["content"]:
                        st.html(image_html)
                elif item_type == "code_input":
                    with st.status("Code", state="complete"):
                        st.code(item["content"])
                elif item_type == "code_output":
                    with st.status("Results", state="complete"):
                        st.code(item["content"])



def display_title_small():
    st.markdown(
        """<h1 style='text-align: center; font-family: monospace; letter-spacing: 2px; font-size: 18px;'>
        research assistant
        </h1>
        """, unsafe_allow_html=True)

# def refine_hypothesis_with_llm(client, hypothesis, instructions):
    
#     response = client.responses.create(
#         model="gpt-4o",
#         tools=[{"type": "web_search_preview"}],
#         input=f"Hypothesis: {hypothesis}. {instructions}",
#         text={
#             "format": {
#                 "type": "json_schema",
#                 "name": "refined_hypotheses",
#                 "schema": REFINED_HYPOTHESES_EXTRACTED_SCHEMA_SIMPLE,
#                 "strict": True
#             }
#         }
#     )
#     return response.output_text


def refine_hypothesis_with_llm(client, message, previous_response_id):
    """
    This function returns 
    """
    if previous_response_id:
        response = client.responses.create(
            model="gpt-4o",
            previous_response_id=previous_response_id,
            input=[{"role": "system", "content": message}],
        )
    
    else:
        response = client.responses.create(
            model="gpt-4o",
            input=[{"role": "system", "content": message}],
        )

    return {"response": response.output_text, "response_id": response.id}


def filter_assistant_messages(page) -> list:
    """
    Given a SyncCursorPage[Message] object, return a list of dicts containing the IDs and full message objects
    for all messages where role == 'assistant'.

    Args:
        page: SyncCursorPage[Message] object with a .data attribute (list of Message objects).

    Returns:
        List[dict]: Each dict has 'id' and 'message' keys for assistant messages.
    """
    filtered = []
    for msg in page.data:
        # msg is a Message object with attributes id and role
        if getattr(msg, 'role', None) == 'assistant':
            filtered.append({
                'id': msg.id,
                'message': msg
            })
    return filtered


def display_hypotheses_in_sidebar(hypotheses_data, client, history):
    """
    Displays each hypothesis in the sidebar with:
      - Title, text
      - A green "Accept the hypothesis" button
      - Chat input to develop hyptheses furter
    Upon button press, the hypothesis is 'accepted' and the final LLM-refined
    output is displayed. The button is disabled for that hypothesis thereafter.
    """
    # We use a dictionary in session state to track which hypotheses are accepted 
    # and store the final refined output from the LLM/web search.
    # if "accepted_hypotheses" not in st.session_state:
    #     st.session_state["accepted_hypotheses"] = {}

    # Inject global CSS to make *all* st.button elements green
    # (Streamlit does not provide a direct color parameter for st.button)
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
            background-color: green !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    hypotheses = hypotheses_data.get("hypotheses", [])

    selected_hypothesis = st.sidebar.selectbox("", options=[hypothesis['title'] for hypothesis in hypotheses])

    selected_hypothesis_title = [hypo["title"] for hypo in hypotheses if hypo["title"] == selected_hypothesis][0]
    selected_hypothesis_text = [hypo["text"] for hypo in hypotheses if hypo["title"] == selected_hypothesis][0]

    st.sidebar.subheader(selected_hypothesis_title)
    st.sidebar.write(selected_hypothesis_text)

    if st.sidebar.button("Accept the refined hypothesis", key=f"accept_btn_{selected_hypothesis}"):
        refined_output = refine_hypothesis_with_llm(
            client,
            message=history
        )
        # Store the refined output so the user can see it in the UI
        st.session_state["accepted_hypotheses"][i] = refined_output
        # Force immediate re-run so the button disappears
        st.rerun()

    prompt = st.chat_input("Discuss the refined hypotheses further or accept it.", key=f'chat_input_{selected_hypothesis}')

    the_thing = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)

    assistant_messages = filter_assistant_messages(the_thing)

    import pprint
    pprint.pprint(assistant_messages)
    # pprint.pprint(f"\n\nTYPE OF THE THING: {type(the_thing[-1])}")
    # pprint.pprint(f"\n\nLIST OF MESSAGES: {the_thing[-1].id}")    

    if prompt:
        refine_hypothesis_with_llm(
            client,
            message=prompt,
            previous_response_id=client.messages.id
        )


    # for i, hypothesis in enumerate(hypotheses_data.get("hypotheses", [])):
    #     sidebar = st.sidebar
    #     sidebar.subheader(hypothesis["title"])
    #     sidebar.write(hypothesis["text"])
        
        
    #     # Check if we've already accepted this hypothesis
    #     if i in st.session_state["accepted_hypotheses"]:
    #         sidebar.success("This hypothesis has been accepted.")
    #         st.subheader(hypothesis["title"])
    #         st.write(st.session_state["accepted_hypotheses"][i])
    #         st.markdown("---")
    #     else:
    #         # Show the button if not yet accepted
    #         if sidebar.button("Improve this hypothesis with LLM and web search", key=f"accept_btn_{i}"):
    #             refined_output = refine_hypothesis_with_llm(
    #                 client, 
    #                 message=hypothesis)
    #             # Store the refined output so the user can see it in the UI
    #             st.session_state["accepted_hypotheses"][i] = refined_output
    #             # Force immediate re-run so the button disappears
    #             st.rerun()