import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000/ask/"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .user-bubble {
        background-color: #d1f7c4;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-bubble {
        background-color: #f0f0f0;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Chatbot Settings")
st.sidebar.markdown(
    """
    **Powered by:**
    - FastAPI (Backend)
    - Streamlit (Frontend)
    """
)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared!")

st.title("RAG Chatbot 🤖")
st.write("Ask me anything!")

query = st.text_input("Your Question:")

if query:
    st.session_state.messages.append({"role": "user", "text": query})

    try:
        response = requests.get(FASTAPI_URL, params={"query": query})
        if response.status_code == 200:
            data = response.json()
            bot_response = data.get("response", "Sorry, I didn't understand that.")
        else:
            bot_response = "Error: Unable to connect to the chatbot backend."
    except Exception as e:
        bot_response = f"Error: {e}"

    
    st.session_state.messages.append({"role": "bot", "text": bot_response})

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{message["text"]}</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-bubble">{message["text"]}</div>', unsafe_allow_html=True
        )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    ---
    **Note:** This chatbot uses a RAG (Retrieval-Augmented Generation) architecture to answer your questions!
    """
)
