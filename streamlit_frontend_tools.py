import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_backend_tools import chatbot, retrieve_all_threads
import uuid

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    st.session_state["message_history"] = []
    st.session_state["thread_id"] = generate_thread_id()
    add_thread(st.session_state["thread_id"])

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# Initialize session state variables
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# the UI
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My conversation")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        # format the loaded messages from the chat history
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                pass

            temp_messages.append({"role": role, "content": msg.content})
        
        st.session_state["message_history"] = temp_messages

# writing all the existing messages in the current conversation UI
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input('Type Here')

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn"
        }

    def stream_content(user_input):
        stream_chunk = chatbot.stream({'messages': [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode='messages')

        for message_chunk, metadata in stream_chunk:
            if not message_chunk.content:
                continue

            # Extract text from list-of-dicts structure
            for part in message_chunk.content:
                if isinstance(part, dict) and "text" in part:
                    yield part["text"]

    with st.chat_message("assistant"):
        ai_message = st.write_stream(stream_content(user_input))
        
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})