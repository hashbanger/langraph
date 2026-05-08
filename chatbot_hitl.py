from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
import os

os.environ["LANGCHAIN_PROJECT"] = "chatbot_hitl"

load_dotenv()

model = ChatOpenAI()

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

@tool
def purchase_tool(symbol: str, quantity: int, stock_price: float) -> dict:
    "Simulating purchasing of a stock."
    decision = interrupt(f"Do you approve the purchase of {quantity} shares of {symbol} at price {quantity * stock_price}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchse order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "declined",
            "message": f"Purchase order for {quantity} shares of {symbol} was declined.",
            "symbol": symbol,
            "quantity": quantity,
        }

tools = [get_stock_price, purchase_tool]
model_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    thread_id = "demo_thread"

    while True:
        user_input = input("User: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state = {"messages": [HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        interrupts = result.get("__interrupt__", [])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"Human: {prompt_to_human}")
            decision = input("Decision: ").strip().lower()

            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}}
            )

        messages = result["messages"]
        last_msg = messages[-1]

        print(f"AI: {last_msg.content}\n")