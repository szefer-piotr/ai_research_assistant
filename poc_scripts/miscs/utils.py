from typing_extensions import override
from openai import AssistantEventHandler
import streamlit as st

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        st.session_state["messages"].append({"role": "assistant", "content": text})