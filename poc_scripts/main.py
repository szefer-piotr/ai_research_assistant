import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAI class
client = OpenAI(api_key=openai_api_key)

assistant = client.beta.assistants.create(
    name="Python Data Analyst",
    instructions="You are a data analyst working with a Python data analysis library. Write code, analyze data to provide insights.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
)


st.title("Research Assistant")
st.subheader("Upload a csv file and analyse it with a coding agent")

uploaded_file = st.file_uploader(
    "Choose a CSV file", type="csv", label_visibility="collapsed"
)

if uploaded_file is not None:
    # Create a status indicator
    with st.status("Starting work...", expanded=False) as status_box:
        # Upload  the file to OpenAI
        file = client.files.create(
            file=uploaded_file,
            purpose="assistants"
        )
        
        # Create a thread with the uploaded file
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "I want to analyze this data"
                 }
            ]
        )

        # Create a run with the thread
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            file_ids=[file.id]
        )

        while run.status != "completed":
            time.sleep(1)
            status_box.update(label=f"{run.status}...", state="running")
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id, 
                run_id=run.id
            )

        status_box.update(label="Completed", state="complete", expanded=True)
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        st.markdown(messages.data[0].content[0].text.value)

        # Delete the uplaoed file from OpenAI

