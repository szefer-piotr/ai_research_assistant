import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

st.set_page_config(page_title="Research Assistant", page_icon="🧠", layout="wide")