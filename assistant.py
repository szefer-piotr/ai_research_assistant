from openai import OpenAI
from dotenv import load_dotenv
import os

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

