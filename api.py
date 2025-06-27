from flask import Flask
from flask_socketio import SocketIO, emit
from agent import graph_builder  # example placeholder
from langchain_core.messages import AIMessage,HumanMessage
from typing import TypedDict
from typing_extensions import Annotated

class message_state(TypedDict):
    msg:list
    user:list
    bot:list
curr_msg: message_state = {
    "msg": [],
    "user": [],
    "bot": []
}
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# You'd replace this with your actual LangGraph logic
def get_bot_response(message):
    # Here you should invoke your LangGraph agent
    result = graph_builder.invoke({"message": message, "user_id": '12342'})
    print("resu",result)
    return result

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    
@socketio.on('user_message')
def handle_user_message(data):
    print(f"User {data['user_id']} says:", data["message"])
    curr_msg['msg'].append(HumanMessage(data['message']))
    print("the user is",curr_msg['msg'])
    response = get_bot_response(curr_msg['msg'])
    curr_msg['msg'].append(AIMessage(response['output']))
    print(curr_msg['msg'])
    
    print(response,"thiadbqwdbqu")
    emit("bot_response", {"message":response['output']})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080,debug=True)

