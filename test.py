from langchain_ollama.chat_models import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

tool_node = ToolNode([get_weather])
base_url="http://172.16.10.2:11434"

model = ChatOllama(model="llama3.3",base_url=base_url)
model_with_tools = model.bind_tools([get_weather])

def should_continue(state: MessagesState):
    messages = state["messages"]
    print(messages,"twjbfjhe2wrbvclewevwvv")
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    print(response,"00000000000000000000000")
    return {"messages": [response]}

builder = StateGraph(MessagesState)

# Define the two nodes we will cycle between
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

graph = builder.compile()
with open("graph1.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())                                                                
    
res = graph.invoke({"messages": [{"role": "user", "content": "what's the weather in sf?"}]})
