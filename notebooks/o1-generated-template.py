import os
import openai
import pandas as pd
from io import StringIO
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# If you want to chunk PDF text, you can also import TextSplitter utilities:
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

################################
# 1. CONFIGURE YOUR OPENAI KEYS
################################

# Make sure to set your OpenAI credentials properly via environment variables
# or any other secure method. For example:
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# In code, you might do:
# openai.api_key = os.getenv("OPENAI_API_KEY")

##############################
# 2. HELPER FUNCTIONS
##############################

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyPDF2.
    """
    text_content = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_content.append(page.extract_text())
    return "\n".join(text_content)

def extract_methodology_section(full_text: str) -> str:
    """
    A naive approach to extract only the 'Methodology' section from
    the full PDF text. Adjust to your own needs. 
    """
    # Example: Find the text from 'Methodology' heading to next heading like 'Results'
    # This is very simplistic and might need to handle text structures properly.
    start_keyword = "Methodology"
    end_keywords = ["Results", "Analysis", "Discussion", "Conclusion"]
    
    start_index = full_text.lower().find(start_keyword.lower())
    if start_index == -1:
        return ""
    
    # Search for the earliest next section heading
    subsequent_indices = []
    for ek in end_keywords:
        idx = full_text.lower().find(ek.lower(), start_index + len(start_keyword))
        if idx != -1:
            subsequent_indices.append(idx)
    
    if not subsequent_indices:
        # If we don't find any subsequent heading, take everything after 'Methodology'
        return full_text[start_index:]
    
    end_index = min(subsequent_indices)
    return full_text[start_index:end_index]

def generate_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary for each column in the dataset:
      - Numeric columns: min, max, mean, std, etc.
      - Categorical columns: unique values, counts
      - Date/Time columns: min date, max date
    Return a dictionary containing the summary data.
    """
    summary_dict = {}
    
    for col in df.columns:
        col_info = {}
        col_data = df[col].dropna()
        
        # Try to convert to datetime - if works, treat as time column
        try:
            col_data_dt = pd.to_datetime(col_data, errors='raise')
            # If conversion successful, assume time column
            col_info["column_type"] = "datetime"
            col_info["time_span_start"] = str(col_data_dt.min())
            col_info["time_span_end"] = str(col_data_dt.max())
        except ValueError:
            # Not date/time, proceed to numeric or categorical logic
            if pd.api.types.is_numeric_dtype(col_data):
                col_info["column_type"] = "numeric"
                col_info["count"] = int(col_data.count())
                col_info["mean"] = float(col_data.mean())
                col_info["std"] = float(col_data.std())
                col_info["min"] = float(col_data.min())
                col_info["max"] = float(col_data.max())
            else:
                col_info["column_type"] = "categorical"
                uniques = col_data.unique()
                col_info["unique_values"] = [str(u) for u in uniques]
                col_info["unique_value_count"] = len(uniques)
        
        summary_dict[col] = col_info
    
    return summary_dict

def dict_to_xml_summarization(methodology_summary: str,
                              statistics_extraction: str,
                              dataset_summary: dict) -> str:
    """
    Create an XML string combining methodology summary, 
    extracted statistical analyses, and dataset summary.
    """
    import xml.etree.ElementTree as ET
    
    root = ET.Element("SummaryOutput")
    
    # Methodology part
    methodology_el = ET.SubElement(root, "MethodologySummary")
    methodology_el.text = methodology_summary
    
    # Statistical analyses part
    stats_el = ET.SubElement(root, "StatisticalAnalyses")
    stats_el.text = statistics_extraction
    
    # Dataset Summary
    data_el = ET.SubElement(root, "DatasetSummary")
    for col, info in dataset_summary.items():
        col_el = ET.SubElement(data_el, "Column", name=col)
        for key, val in info.items():
            sub_el = ET.SubElement(col_el, key)
            if isinstance(val, list):
                sub_el.text = ", ".join(val)
            else:
                sub_el.text = str(val)
    
    # Convert to string
    return ET.tostring(root, encoding='unicode')

##############################
# 3. LANGCHAIN LLM SETUP
##############################

# Define your LLMs. 
# Replace "gpt-4o", "gpt-o1", and "gpt-o1-mini" with the actual model names or 
# endpoints you have configured. 
# For demonstration, we'll just use placeholders in the constructor:

summarization_llm = OpenAI(
    temperature=0.0,       # Low temperature for factual summarization
    model_name="gpt-4o"    # Hypothetical Summarization Model
)

planner_llm = OpenAI(
    temperature=0.0,
    model_name="gpt-o1"    # Hypothetical Planner Model
)

executor_llm = OpenAI(
    temperature=0.0,
    model_name="gpt-o1-mini"  # Hypothetical Executor Model
)

# PROMPTS
methodology_prompt = PromptTemplate(
    input_variables=["methodology_text"],
    template=(
        "You are a model specialized in summarizing methodology sections. "
        "Read the following text delimited by triple backticks and provide:\n"
        "1) A concise summary of the methodology.\n"
        "2) Exactly which statistical analyses were performed.\n"
        "3) How data was obtained.\n"
        "```\n{methodology_text}\n```"
    )
)

planner_prompt = PromptTemplate(
    input_variables=["methodology_summary", "dataset_summary"],
    template=(
        "You are a planning model. Based on the methodology summary and dataset summary, "
        "plan a step-by-step routine (as programmatic pseudocode or structured steps) "
        "to execute the identified statistical analyses on the full dataset.\n\n"
        "Methodology Summary:\n{methodology_summary}\n\n"
        "Dataset Summary:\n{dataset_summary}\n\n"
        "Provide your plan in XML format, with each <Step> containing a structured explanation "
        "of how to implement it programmatically."
    )
)

executor_prompt = PromptTemplate(
    input_variables=["analysis_plan_xml"],
    template=(
        "You are an executor model specialized in generating R scripts for each step. "
        "Given the plan in XML, do the following:\n"
        "1. Generate separate R scripts for each analysis step.\n"
        "2. Generate a single master R script that runs them all in a structured manner.\n"
        "Output your results clearly, indicating how the scripts should be saved.\n\n"
        "Plan XML:\n{analysis_plan_xml}"
    )
)

##############################
# 4. MAIN WORKFLOW
##############################

def main():
    # --- Step 1: Read PDF and extract methodology ---
    pdf_path = "path/to/your/paper.pdf"
    full_text = extract_text_from_pdf(pdf_path)
    methodology_text = extract_methodology_section(full_text)
    
    # --- Step 2: Summarize the methodology section & extract stats info ---
    summarization_chain = LLMChain(llm=summarization_llm, prompt=methodology_prompt)
    summary_result = summarization_chain.run(methodology_text=methodology_text)
    
    # Here, you can parse the result if you need to separate:
    # 1) Short summary
    # 2) Statistical analyses
    # 3) Data acquisition
    #
    # For simplicity, assume the model returns them in a structured textual format.
    # We'll just keep the entire text as `summary_result` or do a simple split if needed.
    
    # You might do advanced parsing or instruct the model to produce JSON for simpler extraction.
    # In this example, let's assume the final LLM text includes all necessary info.
    
    # We'll store them as separate text fields, if the model outputs them in some delineated form:
    # For demonstration, let's pretend the model returns something like:
    # "SUMMARY: <summary> ... </summary>\nSTAT_ANALYSES: <analyses> ... </analyses>\nDATA_OBTAINED: <data> ... </data>"
    # We'll do naive splitting. Adjust to your actual structure:
    
    summary_text = ""
    stat_analyses_text = ""
    data_obtained_text = ""
    
    # A naive parse:
    lines = summary_result.split("\n")
    current_section = None
    for line in lines:
        line_upper = line.strip().upper()
        if line_upper.startswith("SUMMARY:"):
            current_section = "SUMMARY"
            summary_text = line.partition(":")[2].strip()
        elif line_upper.startswith("STAT_ANALYSES:"):
            current_section = "STAT_ANALYSES"
            stat_analyses_text = line.partition(":")[2].strip()
        elif line_upper.startswith("DATA_OBTAINED:"):
            current_section = "DATA_OBTAINED"
            data_obtained_text = line.partition(":")[2].strip()
        else:
            if current_section == "SUMMARY":
                summary_text += " " + line
            elif current_section == "STAT_ANALYSES":
                stat_analyses_text += " " + line
            elif current_section == "DATA_OBTAINED":
                data_obtained_text += " " + line
    
    # --- Step 3: Load dataset and produce summary ---
    # Suppose you have a CSV or XLSX
    data_path = "path/to/your/dataset.csv"  # or .xlsx
    df = pd.read_csv(data_path)  # if XLSX -> pd.read_excel(data_path)
    
    dataset_summary_dict = generate_dataset_summary(df)
    
    # --- Step 4: Create the combined XML summarization ---
    # Combine all partial texts into single string for "statistical analyses" 
    # since we have it from stat_analyses_text and data_obtained_text 
    combined_stats_info = f"Statistical analyses: {stat_analyses_text}\nData obtained: {data_obtained_text}"
    
    final_summary_xml = dict_to_xml_summarization(
        methodology_summary=summary_text,
        statistics_extraction=combined_stats_info,
        dataset_summary=dataset_summary_dict
    )
    
    # You now have an XML containing:
    # 1. Methodology summary
    # 2. Statistical analyses
    # 3. Dataset summary
    
    # Print or save your final XML
    print("=== METHODOLOGY & DATASET SUMMARY XML ===")
    print(final_summary_xml)
    
    # --- Step 5: Request a plan from the Planner Model (gpt-o1) ---
    planner_chain = LLMChain(llm=planner_llm, prompt=planner_prompt)
    
    # We'll pass the summary_text (methodology) and a short version of the dataset summary
    # You could also convert dataset_summary_dict to a textual representation
    dataset_summary_str = str(dataset_summary_dict)  # or build a more user-friendly text
    plan_result_xml = planner_chain.run(
        methodology_summary=summary_text,
        dataset_summary=dataset_summary_str
    )
    
    print("=== ANALYSIS PLAN (XML) ===")
    print(plan_result_xml)
    
    # --- Step 6: Ask Executor Model to generate R code ---
    executor_chain = LLMChain(llm=executor_llm, prompt=executor_prompt)
    executor_result = executor_chain.run(analysis_plan_xml=plan_result_xml)
    
    # The result should contain instructions for multiple R scripts + a master script
    print("=== R CODE GENERATION ===")
    print(executor_result)
    
    # You would then parse `executor_result` to extract each script’s content 
    # and save them to `.R` files. For example:
    # parse out labeled sections like:
    #   <SCRIPT name="step1.R"> ... </SCRIPT>
    #   <SCRIPT name="step2.R"> ... </SCRIPT>
    #   <MASTER_SCRIPT> ... </MASTER_SCRIPT>
    # Then write to disk:
    #
    # with open("step1.R", "w") as f:
    #     f.write(content_of_step1)
    # with open("main_analysis_pipeline.R", "w") as f:
    #     f.write(master_script_content)
    #
    # Depending on how your LLM structure is returning them.
    
    print("=== WORKFLOW COMPLETE ===")

if __name__ == "__main__":
    main()
