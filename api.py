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

socketio = SocketIO(app, cors_allowed_origins="*")

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
        
    print("the user is",curr_state['message'])
    response = get_bot_response(curr_state)
    curr_state['message'].append(AIMessage(response['output']))
    curr_state['next_node'] = response['next_node']
    curr_state['intent'] = response['intent']
    print(curr_state['message'])
    
    print(response,"thiadbqwdbqu")
    emit("bot_response", {"message":response['output']})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080,debug=True)

