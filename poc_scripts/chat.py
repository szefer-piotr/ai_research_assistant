import streamlit as st
from dotenv import load_dotenv
import os
import openai

# Set page config for better aesthetics
st.set_page_config(
    page_title="My Slick Chatbot",
    layout="centered"
)

# ---- Some optional CSS for a 'slicker' look ----
st.markdown(
    """
    <style>
    /* Hide Streamlit main menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat bubbles */
    .user-bubble {
        background-color: #1E88E5;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 70%;
    }
    .assistant-bubble {
        background-color: #F1F3F4;
        color: black;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 70%;
    }
    .assistant-bubble p {
        margin: 0;
    }

    /* Container styling */
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# def chat_stream(prompt):
#     response = f'You said: "{prompt}"'
#     for char in response:
#         time.sleep(0.1)
#         yield char

st.title("Research Assistant")

chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>', 
                unsafe_allow_html=True
            )
        elif msg["role"] == "assistant" and msg["content"]:
            st.markdown(
                f'<div class="assistant-bubble"><p>{msg["content"]}</p></div>', 
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

user_input = chat_container.text_input("How are you?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    response_placeholder = st.empty()

    streamed_answer =  ""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=st.session_state.messages,
        stream=True,
        temperature=0.7
    )

    for chunk in response:
        print("***"*10)
        print(chunk)
        # chunk_content = chunk["choices"][0]["delta"].get("content", "")
        # chunk_content = chunk.choices[0].delta.content
        # chunk_content = chunk.choices[0].delta.get("content", "")
        chunk_content = getattr(chunk.choices[0].delta, 'content', '') or ""
        streamed_answer += chunk_content
        response_placeholder.markdown(f"**Asssitant:** {streamed_answer}")

    st.session_state.messages.append(
        {"role": "assistant", "content": streamed_answer}
    )


# if "history" not in st.session_state:
#     st.session_state.history = []

# for i, message in enumerate(st.session_state.history):
#     with st.chat_message(message["role"]):
#         st.write(message["content"])
#         if message["role"] == "assistant":
#             feedback = message.get("feedback", None)
#             st.session_state[f"feedback_{i}"] = feedback
#             st.feedback(
#                 "thumbs",
#                 key=f"feedback_{i}",
#                 disabled=feedback is not None,
#             )

# prompt = st.chat_input(
#     "How are you?", 
#     accept_file=True,
#     file_type=["csv"])
# if prompt and prompt.text:
#     st.markdown(prompt.text)
# if prompt and prompt["files"]:
#     st.markdown(prompt["files"])
#     df=pd.read_csv(prompt["files"][0])
#     st.write(df)

# with st.chat_message("assistant"):
#     st.write("Lets analyse some data :wave:")

# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback