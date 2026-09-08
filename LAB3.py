import streamlit as st
from openai import OpenAI

st.title("💬 Lab 3 - Chatbot with Memory")
st.write("A streaming chatbot that answers simply and remembers our recent conversation.")

# get the API key from secrets and create the client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# how many exchanges to remember (Part B: keep only the last 2 user+assistant pairs)
BUFFER_EXCHANGES = 2

# Part C: the system prompt controls how the bot behaves.
# It must always be sent, so the buffer must never remove it.
SYSTEM_PROMPT = (
    "You are a friendly assistant talking to a 10-year-old. "
    "Always explain things simply, using easy words a 10-year-old can understand. "
    "After you answer a question, always ask: 'Do you want more info?' "
    "If the user says yes, give more information about the same topic, then ask "
    "'Do you want more info?' again. "
    "If the user says no, say okay and ask what else you can help them with."
)

# set up the message history in session_state so it survives Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! What would you like to know?"}
    ]

# show all the messages we have so far (we don't show the hidden system prompt)
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
    buffered_messages = st.session_state.messages[-(BUFFER_EXCHANGES * 2):]

    # Part C: put the system prompt first so it is never lost by the buffer
    
    messages_to_send = [{"role": "system", "content": SYSTEM_PROMPT}] + buffered_messages

    # ask the model and stream the answer
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)

    # save the assistant's reply so it becomes part of the memory
    st.session_state.messages.append({"role": "assistant", "content": response})