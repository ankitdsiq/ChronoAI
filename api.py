from flask import Flask,request
from flask_socketio import SocketIO, emit
from agent import graph_builder  # example placeholder
from langchain_core.messages import AIMessage,HumanMessage
from typing import TypedDict
from typing_extensions import Annotated
import requests

class message_state(TypedDict):
    intent:str
    message:list
    user_id:str
    output:str
    next_node:str
    payload:dict
    
    
curr_state: message_state = {
    "intent":None,
    "output": "",
    "message": [],
    'user_id' : None ,
    'next_node':None,
    'payload':{}
}

app = Flask(__name__)

# @app.route("/",methods=['POST'])
# def chat():
#     try:
#         payload = request.get_json()
#         user_id = payload['user_id']
#         if user_id != curr_msg["user_id"]:
#             print("hello")
#             curr_msg['user_id']= user_id
#         if curr_msg['intent']!= payload['intent']:
#             curr_msg['intent'] = payload['intent']
#         user_msg = payload['message']
#         curr_msg['user_msg'].append(user_msg)
#         bot_invoke = graph_builder.invoke(curr_msg)
#         print(bot_invoke)
#         return curr_msg
#     except Exception as e:
#         raise e
from langchain.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # to vectorize the data

socketio = SocketIO(app, cors_allowed_origins="*")

def get_shared_chroma():
    return Chroma(
        collection_name="multi_user_chatbot_memory",
        embedding_function=embedding,
        persist_directory="./chroma_db_chronoai_memory",
    )

def persist_the_memory(state:message_state):
    vectorstore = get_shared_chroma()
    print("storing info",state['message'])
    
    docs = []
    
    for msg in state['message']:
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        docs.append(Document(
            page_content=msg.content,
            metadata={"role": role}  
        ))
    print(docs,"uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
        
    vectorstore.add_documents(docs)
    return state

# You'd replace this with your actual LangGraph logic
def get_bot_response(state):
    # Here you should invoke your LangGraph agent
    result = graph_builder.invoke(state)
    print("resu",result)
    return result

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    
@socketio.on('user_message')
def handle_user_message(data):
    print(f"User {data['user_id']} says:", data["message"])
    curr_state['message'].append(HumanMessage(data['message']))
    if curr_state['user_id'] == None:
        curr_state['user_id']=data['user_id']
        
    response = get_bot_response(curr_state)
    print('bot res __________',response)
    curr_state['message'].append(AIMessage(response['output']))
    curr_state['next_node'] = response['next_node']
    curr_state['intent'] = response['intent']
    curr_state['payload'] = response['payload']
    
    emit("bot_response", {"message":response['output']})

@socketio.on('disconnect')
def handle_disconnect():
    print(curr_state['message'])
    persist_the_memory(curr_state)
    print("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080,debug=True)

