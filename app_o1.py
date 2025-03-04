import streamlit as st
import pandas as pd
import time
from dotenv import load_dotenv
from openai import OpenAI
import os
import openai
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LLM CALLS
def llm_summarize_dataset(dataset_description: str) -> str:
    """
    Summarize the provided dataset_description using OpenAI's ChatCompletion.
    You must have your OpenAI API key set as an environment variable named OPENAI_API_KEY,
    or otherwise provide it to openai.api_key.
    """

    # Construct a prompt or message that instructs the model to summarize the dataset
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful data science assistant. The user will provide a textual "
                "description of a dataset, including columns, shapes, sample data, etc. "
                "Your job is to produce a concise, plain-language summary of the dataset."
            )
        },
        {
            "role": "user",
            "content": (
                f"Please summarize the following dataset description:\n\n"
                f"{dataset_description}"
            )
        }
    ]

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",  # or 'gpt-4' if you have access
        messages=messages,
        temperature=0.7,
        max_tokens=200)
        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        # In production, handle/log errors more gracefully
        return f"An error occurred: {str(e)}"



def llm_describe_columns(dataset_sample: pd.DataFrame, metadata: str = "") -> dict:
    """
    Return a dictionary mapping {column_name: description_from_LLM}.
    If metadata is provided, the LLM is instructed to incorporate it
    into the column descriptions. This version attempts to parse JSON
    from the LLM response to ensure structured output.

    NOTE:
        - You must have openai.api_key set, e.g. via an environment variable
          'OPENAI_API_KEY' or by directly assigning openai.api_key = "sk-..."
        - Make sure your package versions are correct for openai and you have
          access to chat-based models like gpt-3.5-turbo or gpt-4.
    """

    # 1) Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # 2) Convert a small sample of the dataset to JSON (or CSV) so the LLM can see representative data
    #    You can also just use df.head().to_dict(orient="records") or something similar
    data_as_json = dataset_sample.to_dict(orient="records")
    sample_str = json.dumps(data_as_json, indent=2)

    # 3) Prepare the system and user messages
    #    - System: Give the LLM context: it’s a data science assistant
    #    - User: Provide instructions, data sample, and optional metadata
    system_message = {
        "role": "system",
        "content": (
            "You are a data science assistant. The user will supply a small sample of a dataset. "
            "You should provide short, plain-language descriptions of each column. If additional "
            "metadata is given, incorporate it appropriately."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"Here is a small sample of the dataset:\n\n{sample_str}\n\n"
            f"Metadata (if any): {metadata}\n\n"
            "Please return a JSON object with each column name as a key and a short textual "
            "description as the value. For example:\n\n"
            '{\n'
            '  "columnA": "Description of columnA",\n'
            '  "columnB": "Description of columnB"\n'
            '}\n\n'
            "If a column name in the dataset is 'Age', an example might be 'column_descriptions[\"Age\"] = \"Age of the individual in years.\"'. "
            "Only respond with valid JSON. Do not include extra commentary."
        )
    }

    # 4) Perform the chat completion
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[system_message, user_message],
            temperature=0.0,        # keep it deterministic for reproducibility
            max_tokens=500
        )

        # 5) Extract the text from the LLM response
        llm_output = response.choices[0].message.content.strip()

        # 6) Attempt to parse the output as JSON
        column_descriptions = json.loads(llm_output)

        # 7) Validate that the result is a dictionary
        if not isinstance(column_descriptions, dict):
            raise ValueError("LLM output is not a JSON object/dictionary.")

        return column_descriptions

    except json.JSONDecodeError:
        # If LLM doesn't provide valid JSON, we might handle it or fallback
        return {
            col: f"LLM failed to provide JSON; fallback description for {col}."
            for col in dataset_sample.columns
        }
    except Exception as e:
        # Handle any other error (e.g., API error, network issues)
        return {
            col: f"An error occurred: {str(e)} (fallback description for {col})"
            for col in dataset_sample.columns
        }


def llm_create_plan(dataset_summary: str, hypotheses: str) -> str:
    """
    Given a summary of the dataset and user hypotheses, create an analysis plan.
    Replace with actual LLM logic (o1).
    """
    # Set your API key from environment variables (or set it directly)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Prepare the system and user messages
    system_message = {
        "role": "system",
        "content": (
            "You are an expert data analysis planner. Your task is to create a clear, structured "
            "analysis plan based on the provided dataset summary and hypotheses. The plan should "
            "outline step-by-step approaches to validate the hypotheses and perform exploratory data analysis."
        )
    }
    
    user_message = {
        "role": "user",
        "content": (
            f"Dataset Summary:\n{dataset_summary}\n\n"
            f"Hypotheses:\n{hypotheses}\n\n"
            "Please generate a detailed, step-by-step analysis plan for the above information."
        )
    }
    
    try:
        # Call the chat completion API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[system_message, user_message],
            temperature=0.7,
            max_tokens=300
        )
        analysis_plan = response.choices[0].message.content.strip()
        return analysis_plan

    except Exception as e:
        return f"An error occurred while generating the analysis plan: {str(e)}"


def llm_refine_plan(current_plan: str, user_input: str) -> str:
    """
    Refine the existing plan with user's feedback using OpenAI's chat-based API.
    Replace with actual LLM logic (o1).
    """
    # Ensure the OpenAI API key is set (e.g., via the OPENAI_API_KEY environment variable)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Define the system message to set context
    system_message = {
        "role": "system",
        "content": (
            "You are an expert data analysis planner. Your job is to refine and improve an existing "
            "analysis plan based on detailed user feedback. The refined plan should incorporate the user's suggestions "
            "and maintain a clear, step-by-step structure."
        )
    }
    
    # Define the user message with the current plan and user feedback
    user_message = {
        "role": "user",
        "content": (
            f"Here is the original analysis plan:\n\n{current_plan}\n\n"
            f"User Feedback:\n{user_input}\n\n"
            "Please provide a refined and improved analysis plan that incorporates the feedback."
        )
    }
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[system_message, user_message],
            temperature=0.7,
            max_tokens=300
        )
        
        refined_plan = response.choices[0].message.content.strip()
        return refined_plan

    except Exception as e:
        return f"An error occurred while refining the plan: {str(e)}"


def llm_generate_code(plan_section: str) -> str:
    """
    Generate code (Python, for data analysis) for a *single step* or portion of the plan.
    Replace with actual LLM logic (o3-mini).
    """
    # In practice, you'd pass the plan section to the model,
    # and it would produce Python code implementing the step.
    placeholder_code = f"# Placeholder code for plan section:\n# {plan_section}\n"
    placeholder_code += "df.head()"  # Just a dummy line as an example
    return placeholder_code

def llm_interpret_report(results_summary: str) -> str:
    """
    Summarize or interpret the final results for the user.
    Replace with actual LLM logic (Summarizer).
    """
    return f"Interpretation of the results:\n{results_summary}"

# -------------------------------
# CODE INTERPRETER-LIKE EXECUTION
# -------------------------------
def execute_code_in_sandbox(code: str, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Simulate code execution in a sandbox or code interpreter environment.
    Return (success, output_message, updated_dataframe).

    For demonstration, we are just going to simulate success or failure,
    and show how you'd handle error loops. Real usage might involve
    calling an API or a separate environment to execute the code safely.
    """
    # **FAKE** code execution example – always success except if code has the word 'ERROR' in it
    if "ERROR" in code:
        return (False, "Simulated error: code contains 'ERROR'.", df)

    # Simulated "execution"
    output_msg = f"Simulated output from executing:\n{code}"

    # If the code modifies df, you could handle that. We'll just pass it back for demonstration.
    return (True, output_msg, df)

def run_code_until_no_error(code: str, df: pd.DataFrame = None, max_attempts: int = 3) -> str:
    """
    Attempt to run the code in a loop, and if there's an error,
    pass the error back to an LLM to correct it. We then re-run
    until success or we hit the max attempt limit.
    """
    attempts = 0
    while attempts < max_attempts:
        success, output_message, df = execute_code_in_sandbox(code, df)
        if success:
            return output_message, df
        else:
            st.error(f"Execution error: {output_message}")
            attempts += 1
            # Provide the error to the LLM to fix
            code = llm_generate_code(
                plan_section=f"Fix the following error:\n{output_message}\nOriginal code:\n{code}"
            )
            st.warning("Retrying with new code...")
    # If we get here, we failed too many times
    return "Failed to execute code after multiple attempts.", df

# -------------------
# STREAMLIT APP LOGIC
# -------------------
def main():
    st.title("LLM-Powered Data Analysis App")

    # ---- STEP 1: Upload data
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV, Excel, or text-based data file", type=["csv", "xlsx", "xls", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
                df = pd.read_excel(uploaded_file)
            else:
                # For .txt or other formats, demonstrate reading with CSV logic or raw text
                df = pd.read_csv(uploaded_file, sep="\t")  # You could adapt this as needed
            st.write("### Preview of the Uploaded Dataset")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("Please upload a file to proceed.")
        st.stop()

    # ---- STEP 2: Summarize the Dataset + Column Descriptions
    st.header("2. Summarize & Describe the Dataset")
    # Minimal numeric summary
    st.write("#### Basic Statistics")
    st.write(df.describe())

    if st.button("Use LLM to Summarize Dataset"):
        # You could pass the entire shape, columns, summary stats, sample rows, etc. to LLM
        dataset_description_str = f"Columns: {df.columns.tolist()}, shape: {df.shape}\nSample:\n{df.head().to_dict()}"
        dataset_summary = llm_summarize_dataset(dataset_description_str)
        st.success(dataset_summary)

        # You can further get column-by-column descriptions
        column_descriptions = llm_describe_columns(df.head(), metadata="")
        for col, desc in column_descriptions.items():
            st.write(f"**{col}**: {desc}")

    # ---- STEP 3: Upload metadata/hypotheses
    st.header("3. Upload Metadata / Hypotheses")
    meta_file = st.file_uploader("Upload a text file with metadata/hypotheses", type=["txt"])
    if meta_file is not None:
        metadata_hypotheses = meta_file.read().decode("utf-8")
        st.write("### Metadata & Hypotheses Content")
        st.write(metadata_hypotheses)
    else:
        metadata_hypotheses = ""
        st.info("No metadata/hypotheses file uploaded yet. You can proceed without it.")

    # ---- (Re)describe columns using metadata, if user wants
    if st.button("Annotate Columns Using Metadata"):
        column_descriptions = llm_describe_columns(df.head(), metadata=metadata_hypotheses)
        for col, desc in column_descriptions.items():
            st.write(f"**{col}**: {desc}")

    # ---- STEP 4: Create an Analysis Plan
    st.header("4. Create an Analysis Plan")
    if st.button("Generate Plan"):
        plan_text = llm_create_plan(dataset_summary="(Your dataset summary here, or from LLM)", 
                                    hypotheses=metadata_hypotheses)
        st.session_state["analysis_plan"] = plan_text  # store in session
    if "analysis_plan" in st.session_state:
        st.write("### Current Plan")
        st.text_area("Analysis Plan", st.session_state["analysis_plan"], height=200)

    # ---- STEP 5: Modify the Plan (Back-and-Forth)
    st.header("5. Refine the Plan")
    user_feedback = st.text_input("Enter your feedback or modifications to the plan:")
    if st.button("Refine Plan"):
        if "analysis_plan" not in st.session_state:
            st.error("No plan found to refine. Generate a plan first.")
        else:
            refined_plan = llm_refine_plan(st.session_state["analysis_plan"], user_feedback)
            st.session_state["analysis_plan"] = refined_plan
            st.success("Plan refined successfully!")

    if "analysis_plan" in st.session_state:
        st.write("### Refined Plan")
        st.text_area("Analysis Plan", st.session_state["analysis_plan"], height=200)

    # ---- STEP 6: Generate Code for Each Step & Execute
    st.header("6. Generate & Execute Code")
    if "analysis_plan" in st.session_state:
        # For demonstration, let's assume we split the plan by lines or by some delimiter
        plan_sections = st.session_state["analysis_plan"].split("\n")

        for i, section in enumerate(plan_sections):
            if not section.strip():
                continue
            if st.button(f"Generate Code for Step {i+1}"):
                code_to_run = llm_generate_code(plan_section=section)
                st.code(code_to_run, language="python")

                # Execute code in a loop until success or max attempts
                with st.spinner("Executing code..."):
                    output_msg, updated_df = run_code_until_no_error(code_to_run, df)
                st.write(output_msg)

                # Potentially update df if code modifies it
                df = updated_df

    # ---- STEP 7: Generate a Result Report
    st.header("7. Generate Final Report")
    if st.button("Create Report"):
        # For demonstration, let's pretend the final results are just a placeholder
        results_summary = "Placeholder final results from the analysis."
        st.write("### Analysis Results")
        st.write(results_summary)

        # Summarize with LLM
        final_interpretation = llm_interpret_report(results_summary)
        st.success(final_interpretation)


if __name__ == "__main__":
    # Initialize session state if needed
    if "analysis_plan" not in st.session_state:
        st.session_state["analysis_plan"] = ""
    main()
