import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
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

st.markdown(
    '''
    ### Upload your data and your hypotheses
    '''
)

## Set the session ststes
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "data_summary" not in st.session_state:
    st.session_state.data_summary = []
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

st.markdown("### Research assistants :tada:")

prompt = st.chat_input("Ask me anything")



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

    sidebar.button("Summarize the data with Assistant")

    if st.session_state.hypotheses:
        sidebar.title("Hypotheses")
        sidebar.markdown(st.session_state.hypotheses[0])
        sidebar.button("Refine hypotheses with Assistant.")
        st.rerun()

    else:
        sidebar.title("Hypotheses")
        uploaded_hypotheses = sidebar.file_uploader("Upload your hypotheses", type=["txt"])
        if uploaded_hypotheses is not None:
            content = uploaded_hypotheses.read().decode("utf-8")
            st.session_state.hypotheses.append(content)
            st.toast("Hypotheses uploaded successfully!")
            st.rerun()



# WHEN FILE IS NOT LOADED YET
if not st.session_state.file_uploaded:
    # Build sidebar menu
    sidebar = st.sidebar
    sidebar.title("Files uploaded.")
    uploaded_file = sidebar.file_uploader("Upload a file", type=["csv", "txt"])

    if uploaded_file is not None:
        # Check if the file has already been uploaded
        if uploaded_file.name in st.session_state.files.keys():
            print(f"Uploaded file name: {uploaded_file.name}")
            st.warning("This file has already been uploaded.")
            st.rerun()

        else: 
            # Display the uploaded file
            df = pd.read_csv(uploaded_file)
            sidebar.dataframe(df)

            # Save the uploaded file in a dictionary
            st.session_state.files[uploaded_file.name] = uploaded_file
            # Change the boolean indicator
            st.session_state.file_uploaded = True
            st.toast("Dataset uploaded successfully")

            # Create a thread for the user whith the uploaded file
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
            st.toast(f"Thread {thread.id} created.")

            print(f"\nThread created with an ID {st.session_state.thread_id}")

            # Rerun the script to move to the next step
            st.rerun()