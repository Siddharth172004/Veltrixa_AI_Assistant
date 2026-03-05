from main import bot
import streamlit as st

st.set_page_config(
    page_title="Veltrixa Dynamics AI Assistant",
    layout="centered"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = "candidate_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Veltrixa Dynamics AI Assistant")

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask Your Query")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        bot_answer = bot(user_query, st.session_state.session_id)

    st.session_state.messages.append(
        {"role": "assistant", "content": bot_answer}
    )

    with st.chat_message("assistant"):
        st.markdown(bot_answer)