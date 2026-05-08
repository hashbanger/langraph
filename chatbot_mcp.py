from langgraph.graph import START, END, StateGraph
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Annotated, TypedDict
from dotenv import load_dotenv

import asyncio

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():
    tools = await client.get_tools()

    model_with_tools = model.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", END)

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Add an expense of 500 INR for the month of 15 April 2026 for books")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())