import os
from openai import OpenAI
from dotenv import load_dotenv
from prompts import EXECUTOR_MESSAGE

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

assistant = client.beta.assistants.create(
    name="Research Assistant",
    instructions=EXECUTOR_MESSAGE,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)

print(f"Created an assistnt with an ID: {assistant.id}")