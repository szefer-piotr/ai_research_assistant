import streamlit as st
import pandas as pd
import json

from utils import (
    client,
    data_summary_assistant,
    code_execution_assistant,
    refine_hypotheses_instructions,
    SIMPLE_SCHEMA,
    REFINED_HYPOTHESES_SCHEMA,
    REFINED_HYPOTHESES_EXTRACTED_SCHEMA,
    handle_stream_events,
    display_messages,
    display_title_small,
    display_hypotheses_in_sidebar
)

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
if "web_search_calls" not in st.session_state:
    st.session_state["web_search_calls"] = []
if "data_processing_output" not in st.session_state:
    st.session_state["data_processing_output"] = []
if "analysis_plan" not in st.session_state:
    st.session_state["analysis_plan"] = {}
if "code_execution" not in st.session_state:
    st.session_state["code_execution"] = {}
if "accepted_hypotheses" not in st.session_state:
        st.session_state["accepted_hypotheses"] = {}

# Track current state of the app
if "app_state" not in st.session_state:
    st.session_state.app_state = "uploading_files"
if "is_processing_done" not in st.session_state:
    st.session_state["is_processing_done"] = False

# if "refinement" not in st.session_state:
#     st.session_state["refinement"] = False
# if "planning" not in st.session_state:
#     st.session_state["planning"] = False


#-------------------------------------------------------------------------------------

if st.session_state.app_state == "uploading_files":
    st.write(st.session_state.app_state)
    # Data uploading logic: upload boxes, data preview.
    st.markdown("""
        <h1 style='text-align: center; font-family: monospace; letter-spacing: 2px;'>
        research assistant
        </h1>
        """, unsafe_allow_html=True)
    st.sidebar.header("Upload Your Files")
    csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
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

            # print(st.session_state.hypotheses)

        except Exception as e:
            st.error(f"Error reading TXT file: {e}")

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
            st.session_state.app_state = "processing"
            print(st.session_state.app_state)
            st.rerun()


if st.session_state.app_state == "processing":
    st.write(st.session_state.app_state)
    # Data processing logic, hide previews, stream code + text.

    if st.session_state["is_processing_done"]:
        with st.sidebar:
            display_title_small()
            if st.session_state.data_processing_output:
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
                button = st.button("Refine the hypotheses with LLM")
                if button:
                    st.session_state.app_state = "refining_hypotheses"
                    st.rerun()

        display_messages(st.session_state.messages)

    elif not st.session_state["is_processing_done"]:
        with st.sidebar:
            display_title_small()
            if st.session_state.data_processing_output:
                button = st.button("Refine the suggested hypotheses with LLM")
                if button:
                    st.session_state.app_state = "refining_hypotheses"
                    st.rerun()

        for file in st.session_state.csv_file:
            openai_file = client.files.create(
                file=st.session_state.csv_file[file],
                purpose="assistants"
                )    
        
        st.session_state.file_ids.append(openai_file.id)
            
        thread = client.beta.threads.create(
            messages=[
                {"role": "user",
                "content": f"""Refine the hypotheses: {st.session_state.hypotheses} using the attached dataset. 
                List the refined hypotheses at the end.
                """,
                "attachments": [{"file_id": st.session_state.file_ids[0],
                                "tools": [{"type": "code_interpreter"}]}]
                }])
        
        st.session_state.thread_id = thread.id
        
        with st.chat_message("assistant"):
            assistant_output = handle_stream_events(
                st, 
                client, 
                data_summary_assistant, 
                st.session_state.thread_id,
                [{"type": "web_search_preview"}]
            )
            # Store the message history manually
            st.session_state.messages.append({"role": "assistant", "items": assistant_output})
            data_description = [text['content'] for text in assistant_output if text["type"] == "text"]
            st.session_state.data_processing_output = data_description

            # Add a message to a thread
            client.beta.threads.messages.create(
                st.session_state.thread_id,
                role="assistant",
                content=data_description,
                metadata={"content_type": "data_description"}
            )

            st.session_state["is_processing_done"]=True
            st.rerun()


elif st.session_state.app_state == "refining_hypotheses":
    st.write(st.session_state.app_state)
    # Data refinement logic, web search, chat with the hypotheses, accept hypotheses
    single_string = "\n".join(st.session_state.data_processing_output)

    # print(f"THE DATA PROCESSING OUTPUT: {single_string}")

    # Extract only the refined hypotheses
    response = client.responses.create(
        model = "gpt-4o-mini",
        input=[
            {"role": "system", "content": "Extract only the refined hypotheses listed from the assistant output."},
            {"role": "user", "content": single_string}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "refined_hypotheses",
                "schema": REFINED_HYPOTHESES_EXTRACTED_SCHEMA,
                "strict": True
            }
        }
    )

    extracted_hypotheses = json.loads(response.output_text)

    ext_hyp = client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="assistant",
        content=response.output_text
    )

    print(ext_hyp.id)

    # Add this to the history
    # history = {"role": "assistant", "content": extracted_hypotheses}

    with st.sidebar:
        display_title_small()
    
    display_hypotheses_in_sidebar(
        extracted_hypotheses, 
        client, 
        history=refine_hypotheses_instructions
        )
    
    if len(st.session_state["accepted_hypotheses"]) == len(extracted_hypotheses['hypotheses']):
        st.session_state.app_state = "analysis_plan_generation"
        st.rerun()


elif st.session_state.app_state == "analysis_plan_generation":
    # Generate plan for each hypothesis and allow to accept and discuss.
    with st.sidebar:
        display_title_small()
    st.write(st.session_state["accepted_hypotheses"])
    # for hypothesis in st.session_state["accepted_hypotheses"][1]:
        # print(hypothesis)
        # st.subheader(hypothesis["title"])
        # st.write(hypothesis['text'])
        # st.markdown("---")

elif st.session_state.app_state == "execution":
    pass

elif st.session_state.app_state == "reporting":
    pass