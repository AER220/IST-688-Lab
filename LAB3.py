import streamlit as st
from openai import OpenAI

st.title("💬 Lab 3 - Chatbot with Memory")
st.write("A streaming chatbot that remembers the last two exchanges of our conversation.")

# get the API key from secrets and create the client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# how many exchanges to remember (Part B: keep only the last 2 user+assistant pairs)
BUFFER_EXCHANGES = 2

# set up the message history in session_state so it survives Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

# show all the messages we have so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# get the next message the user types
if prompt := st.chat_input("Type your message..."):
    # save the user's message and show it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Part B buffer: keep only the last 2 exchanges (2 pairs = last 4 messages)
    # send just that recent slice to the model instead of the whole history
    buffered_messages = st.session_state.messages[-(BUFFER_EXCHANGES * 2):]

    # ask the model and stream the answer
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=buffered_messages,
            stream=True,
        )
        response = st.write_stream(stream)

    # save the assistant's reply so it becomes part of the memory
    st.session_state.messages.append({"role": "assistant", "content": response})