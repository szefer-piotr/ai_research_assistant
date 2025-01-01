import streamlit as st
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

# from pandasai.llm.google_gemini import GoogleGemini
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe

# from keys import GOOGLE_GEMINI_API_KEY, OPENAI_API_KEY

openai_api_key = os.getenv('OPENAI_API_KEY')
# print(openai_api_key)

def chat_with_csv(df, prompt):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = SmartDataframe(df, config={"llm": llm, "verbose": True})
    if prompt:
        try:
            result = pandas_ai.chat(prompt)
            print(result)
        except KeyError as e:
            print(f"Error: {e}. Unable to retrieve result.")
    else:
        print("Please enter a prompt.")
    
    return result

st.set_page_config(layout="wide")

st.title("Chat With Your Data")

input_csv = st.file_uploader("Upload your CSV file.", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([1,1])
    with col1:
        st.info("CSV Uploaded successfuly")
        data = pd.read_csv(input_csv, encoding='unicode_escape')
        st.dataframe(data)

    with col2:
        st.info("Chat with our data")
        input_text = st.text_area("Enter your query")
        if input_text is not None:
            if st.button("Submitt"):
                st.info("Your query: "+input_text)
                result = chat_with_csv(data, input_text)
                st.success(result)



# file_path = "./data/dataset.csv"
# df = pd.read_csv(file_path)

# prompt = input("Enter your prompt: ")

# # llm = GoogleGemini(api_key=GOOGLE_GEMINI_API_KEY)
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)
# pandas_ai = SmartDataframe(df, config={"llm": llm, "verbose": True})

# if prompt:
#     try:
#         result = pandas_ai.chat(prompt)
#         print(result)
#     except KeyError as e:
#         print(f"Error: {e}. Unable to retrieve resutl.")
# else:
#     print("Please enter a prompt.")
