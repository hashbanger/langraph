import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input('Type Here')

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.text(user_input)

    def stream_content(user_input):
        stream_chunk = chatbot.stream({'messages': [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode='messages')

        for message_chunk, metadata in stream_chunk:
            if message_chunk.content:
                yield message_chunk

    with st.chat_message("assistant"):
        ai_message = st.write_stream(stream_content(user_input))

        # Generator - comprehension approach
        # ai_message = st.write_stream(
        #     message_chunk.content for message_chunk, metadata in chatbot.stream(
        #         {'messages': [HumanMessage(content=user_input)]},
        #         config= {'configurable': {'thread_id': 'thread-1'}},
        #         stream_mode= 'messages'
        #     )
        # )
        
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})