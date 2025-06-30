from langchain.tools import Tool, tool
from langchain.agents import initialize_agent
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain.schema import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, MessagesState, StateGraph,END,Graph
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import requests

from Rag_test import RAG
from langchain.vectorstores import Chroma
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from typing import TypedDict
from langgraph.graph import StateGraph, END
import datetime
from langchain_core.messages import convert_to_messages
from elevenlabs.client import ElevenLabs
from langchain.tools import tool
import os

import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from elevenlabs import play,VoiceSettings
import io
import whisper

import sounddevice as sd
eleven_lab_client = ElevenLabs(api_key="sk_4c17caa48cf53702ea9a291bf91a61c126df611f9d3fc116")
base_url="http://172.16.10.2:11434"

access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTc1MTI5MDk4NSwianRpIjoiODliNDYyYTAtMmY3ZC00MjU1LWFlZWMtMzBiZmM5ZWQ2NDNiIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IjExNDJjMjAzLWE1YjgtNDE2MC05ZDBmLWU4NTg2NDg0Y2I2MiIsIm5iZiI6MTc1MTI5MDk4NSwiY3NyZiI6ImNmYzQxM2E2LWNkMTItNDkyMy1hNmIzLTYwYzJiNzI5NzFlZCIsImV4cCI6MTc1MTI5Mjc4NSwiaWQiOiIxMTQyYzIwMy1hNWI4LTQxNjAtOWQwZi1lODU4NjQ4NGNiNjIiLCJmaXJzdF9uYW1lIjoiQW5raXQiLCJsYXN0X25hbWUiOiJQYW5kZXkiLCJlbWFpbCI6InF3ZXJ0eUBtYWlsLmNvbSIsIm9yZ2FuaXphdGlvbl9pZCI6IjMyZTU0ZDU5LTMxN2QtNDk4OC04MWU1LTFkODZlNjlmNzJmOSIsIm9yZ2FuaXphdGlvbl9uYW1lIjoiYXNwc29sbiIsInJvbGUiOiJhZG1pbiJ9.NhaqKhXKnVT2IFZ6Z8OCRHiavsp-pLSV_BoyRa9cbnOn2Glq8dpVWDhEQGs1FqcHH3LcyGdrht4t2dY2dgZwUJHc79XK_faMVVb8pZpcXLwvNqmLHaSNNjfbGbPtc-enBPf1vcFpmbYDpPczsADzsdwsS3f-xUdp4ys_gGFPaLyR2g2nu0SLaNJnYMVU0WQmo5LCI2RI4n8CBcDwvOtk8kXkegceXakRLxtecUsuRY0F8Sq73T9OInbIK_K2l3cFZSe05p_6ifjOe7_TFnrB5hfvuvKA92njpWaMEeC-LKq6sEqkAXjNxDAzgv0zAHB2Z6XT1ulYg2Eyf2sCnijsvoFE9lTPvAlROIQYgYW4WsajbRrV4bLRc9oyevin7J85PfdvZvQI7hQFN4ayt1N9pHrVkwLnMLFRm5iMlg1VS_TZcTKe-ycxNHBJzvrHz0na94Ra-jAERrhCVqqJ5JSLs2VTGb_enBAOhKzYZukt3nz2IX2t6DHwaPpBg_25-P-prL30DBxtzzwvFH4PXGhkASDXE_Rr3lhMaUaWxxopBtGlhAUGuo1WFS8q585ZdgxtVcD8evPFLOdfoEBC2CX3boJgg_kL05iFBnnGjlK1rGI45uRHaLR_KdaOnJaAQJ-VTAgr0bn6UbMoK817-E6rgQvQr1OLmebIJyLR1bvrkV4"

headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"  # optional, depends on API
        }

# This is input and output structure fro all nodes
class AgentState(TypedDict):
    intent:str = None
    output: str =None  # The output from that node
    message: list  # The user original msg
    user_id : str   # unique id for identifying user
    next_node:str
    payload:dict

curr_state :AgentState={
    'intent':None,
    'output': str , # The output from that node,
    'message': list,  # The user original msg,
    'user_id' : str  ,
    'next_node':None,
    'payload':{}
    
}

manual_reg = 0

inp = str()  # User query/msg 

# def open_link(state: AgentState):
#     """Opens a given URL in the default web browser."""
#     os.system(f"xdg-open {'http://172.16.10.8:5173/company/register'}")
#     print("please open the link in the browser",http://172.16.10.8:5173/company/register"")
#     return state


import sounddevice as sd

# for i, dev in enumerate(sd.query_devices()):
#     if dev['max_input_channels'] > 0:
#         print(f" Found input device: {dev['name']} at index {i}")



embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # to vectorize the data

# Shared ChromaDB collection for all users
def get_shared_chroma():
    return Chroma(
        collection_name="multi_user_chatbot_memory",
        embedding_function=embedding,
        persist_directory="./chroma_db_chronoai_memory",
    )



# This function used to store the chats into database(Chromadb)
def persist_the_memory(state:AgentState):
    vectorstore = get_shared_chroma()
    vectorstore.add_texts([state['message']], metadatas=[{"user_id": state['user_id']}])
    return 

def load_the_memory(state: AgentState):
    print("loading the memory")
    try:
        con = ""
        for j in state['message']:
            con = con +" "+ j.content
        vectorstore = get_shared_chroma()
        print(con,"jjjjjjjjjjjjjjjjjjjjjj")
        
        prev_memory = vectorstore.similarity_search(
            con,  # user input give for retriving similar data from db
            k=10,     # load last 100 messages
            filter={"user_id": state['user_id']}
            )        
        return prev_memory
    except Exception as e:
        return f"error in prev memory {e}"

user_histories = {}  # temporary store the chat.

# This will give the to 50 chat of user 
def get_user_memory(user_id):
    print("Fetching data from vector store............")
    
    try:
        
        if user_id not in user_histories:
            # Create new in-memory message history
            print("in_memory created....")
            chat_history = ChatMessageHistory()

        # Load messages from Chroma
            vectorstore = get_shared_chroma()
            results = vectorstore.similarity_search(
                inp,  # user input give for retriving similar data from db
                k=10,     # load last 100 messages
                filter={"user_id": user_id}
            )

            # Sort messages chronologically if you saved timestamp
            results.sort(key=lambda r: r.metadata.get("timestamp", 0))

            for doc in results:
                content = doc.page_content
                role = doc.metadata.get("role")  # 'human' or 'ai'
                if role == "human":
                    chat_history.add_user_message(content)
                elif role == "ai":
                    chat_history.add_ai_message(content)

            user_histories[user_id] = chat_history
                
        return user_histories[user_id]
    except Exception as e:
        return f"exception arises {e}"
    
# ************************************** ask like human **********************************
def ask_user(query:str,value:str):
    prompt = PromptTemplate(
        template="""
        You are ChronoAI a friendly assistent for Timechronos.
        You are given query asked by user : {query} 
        You have to ask for value : {value} in friendly way to user
        
        Do not make result too long. Just put it sort and formal
        """,
        input_variables=['query','value']
    )
    parser = StrOutputParser()
    llm = ChatOllama(model='llama3.3:latest',base_url=base_url)
    chain = prompt | llm | parser
    
    return chain.invoke({'query':query,'value':value})
    
    
    
# ******************** Voice assistant **********************

def record_audio_untill_stop(state: AgentState):
    """Record audio from microphone untill enter is pressed"""
    print("recording audio ==========")
    audio_data =[]
    recording = True
    sample_rate = 44100

    
    def record_audio():
        nonlocal audio_data,recording
        with sd.InputStream(samplerate=sample_rate,channels=1,dtype='int16',device=6) as stream:
            print('Recording yourt instructions -----------')
            while recording:
                audio_chunks, _ = stream.read(1024)
                audio_data.append(audio_chunks)
                
    def stop_recording():
        """wair the user input to stop the recording"""
        input()
        nonlocal recording
        recording  =    False
    
    # start recording in a seprate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    
    # start thread to listen enter key
    stop_thread = threading.Thread(target=stop_recording)
    stop_thread.start()
    
    #  wait for both thread to complete
    stop_thread.join()
    recording_thread.join()
    
    # stack all audio cunks to one numpy array
    audio_data = np.concatenate(audio_data,axis=0)
    if audio_data.ndim == 2 and audio_data.shape[1] == 2:
        print("Stereo detected, converting to mono...")
        audio_data = audio_data.mean(axis=1).astype(np.int16)

    
    #  convert to WAV format in memory
    audio_bytes = io.BytesIO()
    write(audio_bytes,sample_rate,audio_data)  #used scipy's write func to save to BytesIO
    audio_bytes.seek(0)
    with open("audio.wav", "wb") as f:
        f.write(audio_bytes.read())
        
    model = whisper.load_model("base")
    result = model.transcribe("audio.wav")
    state['message'] = result['text']
    print(result['text'])
    return state

# ************************* text to voice ***********
def play_audio(msg):
    
    
    res = eleven_lab_client.text_to_speech.convert(
        voice_id = "pNInz6obpgDQGcFmaJgB",
        output_format = "mp3_22050_32",
        text = msg,
        model_id ="eleven_turbo_v2_5",
    )
    play(res)
    return msg

    
    # ***************************************Decide the Intent of user message *******************************************
def decide_the_intent_of_query(input:AgentState) :
    print(" [TOOL] decide_the_intent_of_query called with input:", input["message"],"and intent is",input['intent'])
    print()
    if input['intent']!=None:
        return {
        **input,
        'intent':input['intent'],
        "message":input['message'],
        "output":input['output'],
        "user_id" :input['user_id']
    }
        
        
    
    prompt = PromptTemplate(
    template="""
    You are ChronoAI — a highly accurate intent classification agent for **TimeChronos**, an enterprise software that manages:

    - Timesheet creation and logging
    - Project and task assignments
    - Client project summaries
    - Employee performance reports
    - Billable hour tracking
    - Revenue monitoring

    Your job is to analyze the **user message** and classify it into one of the predefined intents.

    ---

    ### INTENT CATEGORIES (Respond with ONLY one of these in JSON format):
    - "normal_conversation"
    - "exit"
    - "register_org"
    - "login"
    - "timesheet_creation"
    - "RAG QUERY"
    - "create_client"
    - "create_task"
    - "create_project"
    - "performance"
    - "fill_timesheet"
    - "NONE"

    ---

    ### INSTRUCTIONS — FOLLOW STRICTLY:

    1. If the message is a casual or generic chat (e.g., greetings, small talk, identity questions like "who are you", "what's my name"), respond with: **{{"response":"normal_conversation"}}**

    2. If the user expresses a desire to **exit or quit the chat** (e.g., "exit", "quit"), respond with: **{{"response":"exit"}}**

    3. If the user asks **about themselves** (e.g., "what is my email", "show my ID"), treat it as **normal_conversation**

    4. If the user asks for **information about TimeChronos  that must have be fetched from TimeChronos documentation (e.g., "what is TimeChronos", "steps for registration", "features of TimeChronos")***, respond with: **{{"response":"RAG QUERY"}}**

    5. If the user wants to **register a company or organization**, respond with: **{{"response":"register_org"}}**

    6. If the user wants to **login** (e.g., "login me", "log in to my account"), respond with: **{{"response":"login"}}**

    7. If the user wants to **create or log a timesheet**, respond with: **{{"response":"timesheet_creation"}}**
    
    8. If the user wants to **create a client **, respond with: **{{"response":"create_client"}}**
    
    9. If the user wants to **create a project **, respond with: **{{"response":"create_project"}}**
    
    10. If the user wants to **create a task **, respond with: **{{"response":"create_task"}}**
    
    11. If the user wants to **fill the timesheets **, respond with: **{{"response":"fill_timesheet"}}**
    
    12. If the user wants to **generate the summary or report  **, respond with: **{{"response":"performance"}}**

    13. If none of the above apply or the intent is unclear, respond with: **{{"response":"NONE"}}**

    ---

    ### CRITICAL RULE:
    Your response MUST be a valid JSON object from this list only:  
    ["exit", "RAG QUERY", "normal_conversation", "register_org", "NONE", "login", "timesheet_creation","create_client","create_project","create_task",'fill_timesheet','performance']

    Do not generate anything outside of this format.
    `
    ***Descide intent with highly diggerenting betweeen message is query or instruction like how to create timeshhet : RAG QUERY while create timeshhet its intent is create_timesheet *****

    ---

    User message: {message}
    """,
    input_variables=["conversation",'message'],
    )
    
    model1 = ChatOllama(model='llama3.3:latest',base_url=base_url)
    parser1 = JsonOutputParser()
    chain = prompt | model1 | parser1
    

    result = chain.invoke({'message':input['message']})
    
    if result['response'] not in ['exit','RAG QUERY','normal_conversation','register_org','NONE','login','timesheet_creation','create_client','create_project',"create_task",'fill_timesheet','performance']:
        result = 'fall_back'

    print("Intent decided:", result['response'])
    
    return {
        'intent':result['response'],
        "message":input['message'],
        "output":input['output'],
        "user_id" :input['user_id']
    }
# **************************************Intent decided ****************************************************


# ********************************* Normal conversation Module starts **********************************
def normal_conversation(state: AgentState):
    print("[TOOL] Normal connversaton started ")

    model_normal_chat = ChatOllama(model='llama3.3:latest',base_url=base_url)
    parser_nomal_chat =StrOutputParser()

    user_id = state['user_id']
    
    conv = load_the_memory(state=state)
    print(conv,"the conv is ssssssssssssssssss")
    conversation = ""
    for doc in conv:
        conversation+=doc.page_content
        
    prompt = PromptTemplate(
    template=
            '''
        You are **ChronoAI**, a friendly and intelligent chatbot assistant for **TimeChronos** — the software that helps users manage:

        - Timesheet filling
        - Project creation
        - Task assignment
        - Employee performance reports
        - Client summaries
        - Billable hours and revenue tracking

        ---

        ### CONTEXT:
        You are given the **past user conversation**:  
        {conversation}

        The **latest user message** is:  
        {message}

        ---

        ### YOUR GOAL:
        Respond to the **latest message** in a natural and helpful tone — like a trusted colleague or friend — while respecting the following rules:
        - if the query is out of timechronos context then simply do not reply the answer.Simply ask fro timechronos related query

        ---

        ### STRICT INSTRUCTIONS:

        1. **Engage naturally** — avoid robotic responses. Be helpful, concise, and clear.

        2. If the user says something like "bye", "exit", "quit", "goodbye", **do not continue** — just reply with exactly:  
        `"exit"`

        3. **Avoid repeating the user's previous messages** in your response unless absolutely necessary.
        

        ---

        Your job is to reply only to the **latest message** based on the context of the conversation. Think carefully, keep it human, and follow the rules above.
        3. If not needed do not give the previous message in response
        
        
       ***** If user ask anything else timeschronos then Just reply politely that you are chronoai a timechronos assistent. You dont know about his question*****
            ''',
        input_variables=["conversation",'message']
        )
    
    chain = prompt | model_normal_chat  | parser_nomal_chat
        
    res = chain.invoke({'conversation':conversation,'message':state['message']})

    # Add message to vector store manually with user_id metadata
    vectorstore = get_shared_chroma()
    vectorstore.add_texts(str(state['message']), metadatas=[{"user_id": user_id}])
    print(res)

    return {
        **state,
        'intent':None,
        "output": res,
        "message" : state['message'] + [res],
        "user_id":user_id
    }
    
# ************************************************************** Normal conversation Module ended ******************************************************

# *************************************************************** Registration Module starts ************************************************************
def ask_for_manual_reg(state:AgentState):
    if not state['next_node']:
        return{
            **state,
            'output':'do you want manual registration? Yes/No ?',
            'next_node':'manual_registration'
        }
    else:
        return {
            **state,
            'output':'bot_registration'
            
            
        }

def manual_registration(state: AgentState):
    print("entring in manual regisstration...",state['message'][-1])
    answer = state['message'][-1].content.lower()
    is_manual = answer in ['yes', 'y', 'ok']
    return {
        'intent': None,
        'message': state['message'],
        'user_id': state['user_id'],
        'output': 'please open the link in browser' if is_manual else 'bot_registration',
        'next_node': 'manual_link' if is_manual else 'bot_registration'
    }



def extract_information_for_register(state: AgentState):
    print("Input to extract_information_for_register:", state["message"])

    model2 = ChatOllama(model='llama3.3:latest',base_url=base_url)
    parser2 = JsonOutputParser()
    conv = load_the_memory(state=state)
    conversation = ""
    for doc in conv:
        conversation+=doc.page_content
    
    prompt = PromptTemplate(
        template="""
        Extract the following fields [organization_name, email, password, first_name, last_name] from current conversation between AI and user : {state} and the previous conversation {conversation}
        Return the extracted informaton just like : {{"organization_name":"name_of_org","email":"email","password":"12343","first_name":"james","last_name":"kumar"}}
        If you are not able to fetch any field then set the value of that field as "NO" only
        You have to return the output in "Json" format only
        """,
        input_variables=["state","conversation"],
    )

    chain = prompt | model2 | parser2
    try:
        payload = chain.invoke({"state": state['message'],"conversation":conversation})        
        
        while 'NO' in dict(payload).values():
            val = get_all_reg_parameters(dict(payload),state)
            if val == "error in getting reg parameters":
                return {
                    **state,
                "message":state["message"],
                "output":"payload can not be extracted",
                "user_id" :state['user_id']
            }
            return {
                **state,
                'output':val
            }

        vectorstore = get_shared_chroma()
        vectorstore.add_texts([str(state['message'])], metadatas=[{"user_id": state['user_id']}])
        vectorstore.add_texts([str(payload)],metadatas=[{"user_id": state['user_id']}])
        print("payload is ",payload)
        
        return {
            **state,
            "message":state["message"],
            "payload":payload,
            "user_id" :state['user_id']
        }
    except Exception as e:
        return {
            **state,
            "message":state['message'],
            "output":f"payload can not be extracted because {e}",
            "user_id" :state['user_id']
        }
            
    
def get_all_reg_parameters(payload,state) :
    print('TOOL getting all registration parameters........')
    try:
        print("getting all_reg_parameters")
        for key in payload.keys():
            if payload[key]=='NO':
                val = ask_user("to register please provide",f"please provide {key} value ")
                return val
        return 'None'
        
        
    except Exception as e:
        return "error in getting reg parameters"


def generate_review(state: AgentState):
    if state['output'] != "payload can not be extracted":
        print(f"Please review the details fro registration :  {state['output']} ")
        edit = input(ask_user("Please review the details fro registration :  {state['output']}","if need editing please type Y else N"))
        while edit != "N":
            print(f"Please review the details fro registration :  {state['output']} ")
            edit = input(ask_user("Please review the details fro registration :  {state['output']}","if need editing please type Y else N"))
            if edit != "N":
                key,value = input("please provide the detail AS KEY VALUE:").split(" ")
                state['output'][key] = value
        
        return state


def register_the_client(state: AgentState) -> str:
    print("TOOL register_the_client CALLED WITH INPUT:", state)
    try:
        payload= state["payload"]
        
        url = "http://172.16.10.13:5000/register"

        response = requests.post(url, json=payload)        
        
        vectorstore = get_shared_chroma()
        vectorstore.add_texts([f"Human_message is {state['message']}"], metadatas=[{"user_id": state['user_id']}])
        print(response.json(),"wsdfnweuifbcwequvbchue")
    
        return {
            **state,
            "output": response.json()['message'],
            "message" : state["message"],
            "user_id":state["user_id"],
            "payload":{}
        }
    except Exception as e:
        return {
            "output": f"Registration failed with exception {e}",
            "message" : state["message"],
            "user_id":state["user_id"],
        }
        
# *******************************************************************Registration Module ended *****************************************************

# *********************************************************** Login Module start **************************************************************

def extract_the_login_parameters(state:AgentState):
    print("[TOOL] Login parameter extracting from",state)
    model_login = ChatOllama(model='llama3.3:latest',base_url=base_url)
    parser_login  =JsonOutputParser()
    
    conv = load_the_memory(state=state)
    memory = ""
    for doc in conv:
        memory+=doc.page_content
        
    
    prompt_login  =PromptTemplate(
        template="""
        You are Timechronos chatbot. Your task is to extract only the login information in the form of {{email, password}} from the current {query}. 

            If either or both of the values are not present in the {query}, then check the {conversation} for the missing information.

            Instructions:
            - Only extract values for `email` and `password` latest from conversation.
            - If either value is not available in both {query} and {conversation}, respond that the variable is not given.
            - Your final output must be in strict JSON format.

            Example output:
            {{
            "email": "user@example.com",
            "password": "userpassword123"
            }}

            If a variable is missing, e.g.:
            {{
            "email": "user@example.com",
            "password": "not given"
            }}
        
        """,
        input_variables={"query","conversation"}
    )
    
    chain = prompt_login | model_login | parser_login
    
        
    print(state['message'][-1].content,"wqdddddddd")
    doc = Document(metadata={'user_id':state['user_id']},page_content=state["message"][-1].content)
    res = chain.invoke({'query':doc,"conversation":memory})
    return {
        **state,
            "payload": res,
            "message" : state["message"],
            "user_id":state["user_id"],
        }
        

def login_user(state: AgentState):
    print("TOOL login_user  WITH INPUT:", state)
    try:
        # This assumes `input` is already a dict
        payload= state["payload"]
        if not all(k in payload for k in [ "email", "password"]):
            return {
                **state,
                'output':"Missing required fields. Please provide all necessary information."
                }
        url = "http://172.16.10.13:5000/login"
        

        response = requests.post(url, json=payload)
        print(response.json())
        
        return {
            **state,
            "output": f"{response.json()['access_token']}",
            "message" : state["message"],
            "user_id":state["user_id"],
            'payload' :{}
        }
    except Exception as e:
        return {
            **state,
            "output": f"login failed with exception {e}",
            "message" : state["message"],
            "user_id":state["user_id"],
        }

# *********************************************************** Login module ended ****************************************************************

# *********************************************************** CREATE CLIENT ********************************************************
def create_client(state: AgentState):
    try:
        prompt = PromptTemplate(
            template="""
                Extract the client's name and website and project name for the client from the following message: {message}

                Instructions:
                - If any field is not given, set its value to "NO".
                - Return the result strictly in **valid JSON format** with keys: "name" and "website","project_name".
                - Do **not** include any explanation, text, markdown, or additional formatting.

                Example output:
                {{"name": "akp", "website": "https://akp.com",project_name:<extracted_projet_name>}}
            """,
            input_variables=['message']
        )
        parser = JsonOutputParser()
        model = ChatOllama(model='llama3.3:latest',base_url=base_url)
        chain = prompt | model | parser
        
        payload = chain.invoke({'message':state['message']})
        print("clien information ",payload)
        
        url  ="http://172.16.10.13:5000/add-client"
        
        for key in dict(payload).keys():
            if payload[key]=="NO":
                val = ask_user(state['message'],key)
                return {
                    **state,
                    'output':val
                }
                            
        response = requests.post(url,json=payload,headers=headers)
        response =response.json()
        print("response is ",response)
        # if response['message'] == "Record added successfully":
        #     state['output']=response['message']
            
        #     if msg.lower() in  ['yes','y','yup','yaa','ok']:
        #         name = input(ask_user('want to create project for the client registered','project_name'))
        #         description = input("please provide description we recommend it ")
        #         if not description:
        #             description = None
                    
        #         payload= {
        #             'name':name,
        #             'description':description,
        #             'client_id':response['data']['id']
        #         }
                
        #         project_response = requests.post("http://172.16.10.13:5000/add-project",json=payload,headers=headers)
        #         project_response = project_response.json()
                
        #         msg = input("Do you want to add task for this project? : ")
                
        #         if msg.lower() in  ['yes','y','yup','yaa','ok']:
        #             task_name = input(ask_user('to add task for the project',"provide the task name"))
        #             project_id = project_response['data']['id']
        #             url  ="http://172.16.10.13:5000/add-tasks"
        #             response = requests.post(url,json={'name':task_name,'project_id':project_id},headers=headers)
        #             response =response.json()
                    
        #             return{
        #                 'message':state['message'],
        #                 'output':response['message'],
        #                 'user_id':state['user_id']
        #             }
                    
        #         return{
        #                 'message':state['message'],
        #                 'output':project_response,
        #                 'user_id':state['user_id']
        #             }
        return{
            **state,
            'intent':None,
            'message':state['message'],
            'output':response['message'],
            'user_id':state['user_id'],
            'payload':{}
        }
    
    except Exception as e:
        return{
            **state,
            'intent':None,
            'message':state['message'],
            'output':f"exception {e}",
            'user_id':state['user_id']
        }



# *********************************************** Create the project ********************************************************
def create_project(state: AgentState):
    try:
        print("Creating project...................")
        url = "http://172.16.10.13:5000/get-all-clients"
        clients = requests.get(url,headers=headers)
        clients = clients.json()
        if not clients:
            return{
                        'message':state['message'],
                        'output':"You have no client. Please create client",
                        'user_id':state['user_id']
                    }
            
        client_info = dict()
        for i in range(len(clients['clients'])):
            client_info[clients['clients'][i]['name']] = clients['clients'][i]['id']
            
        client_name = input(ask_user(state['message'],f"You have there clients {client_info.keys()} select one to create project"))
        while client_name not in client_info.keys():
            client_name = input(ask_user(state['message'] + f"you have choosen {client_name} which is not registered",f"You have there clients {client_info.keys()} select one to create project"))
            
        selected_client_id = client_info[client_name]
             
        prompt = PromptTemplate(
            template="""
           Extract the project name from the following message: {message}

            Instructions:
            - If the project name is not mentioned, set its value to "NO".
            - Return the result strictly in valid **JSON format** with the key: "name".
            - Do **not** include any explanation, text, markdown, or symbols.

            Example output:
            {{"name": "Project Phoenix"}}

            """,
            input_variables=['message']
        )
        parser = JsonOutputParser()
        model = ChatOllama(model='llama3.3:latest',base_url=base_url)
        chain = prompt | model | parser
        
        payload = chain.invoke({'message':state['message']})
        payload['client_id'] = selected_client_id
        print("project information ",payload)
        
        url  ="http://172.16.10.13:5000/add-project"
        
        for key in dict(payload).keys():
            if payload[key]=="NO":
                payload[key] = input(ask_user(state['message'],f"Hey! provide {key} for project"))
        
        
        project_response = requests.post(url,json=payload,headers=headers)
        project_response =project_response.json()
        if project_response['message'] == "Record added successfully":
                msg = input("Do you want to add task for this project? ")
                
                if msg.lower() in ['yes','y','yup','yaa','ok']:
                    task_name = input(ask_user('want to add task for the project',"provide the task name"))
                    project_id = project_response['data']['id']
                    url  ="http://172.16.10.13:5000/add-tasks"
                    task_reponse = requests.post(url,json={'name':task_name,'project_id':project_id},headers=headers)
                return{
                        'message':state['message'],
                        'output':task_reponse.json()['message'],
                        'user_id':state['user_id']
                    }
                
                
                
        return{
            'message':state['message'],
            'output':project_response['message'],
            'user_id':state['user_id']
        }
        
        
        
    except Exception as e:
        return{
            'message':state['message'],
            'output':f"exception {e}",
            'user_id':state['user_id']
        }
        
# *************************************************************** Create task *********************************************************
def create_task(state: AgentState):
    try:
        print("Creating task...................")
        url = "http://172.16.10.13:5000/get-project-by-client"
        projects = requests.get(url,headers=headers)
        projects = projects.json()
        if not projects:
            return{
                        'message':state['message'],
                        'output':"You have no project. Please create client",
                        'user_id':state['user_id']
                    }

            
        projects_info = dict()
        for i in range(len(projects['projects'])):
            projects_info[projects['projects'][i]['name']] = projects['projects'][i]['id']
            
        prompt = PromptTemplate(
            template="""
                Extract the **task name** and **project name** from the following message: {message}

                Instructions:
                - Ignore command phrases like "create task", "create_task", "add task", etc. They are not part of the task name.
                - Only extract the actual task name and project name mentioned **after** or separate from those command words.
                - If either field is missing, set its value to "NO".
                - Respond strictly in **valid JSON format** with the following keys:
                - `"name"` — for the task name
                - `"project_name"` — for the project name
                - Do **not** include any explanation, markdown, or extra text.

                Example output:
                {{"name": "Fix dashboard bug", "project_name": "Analytics UI"}}

            """,
            input_variables=['message']
        )
        parser = JsonOutputParser()
        model = ChatOllama(model='llama3.3:latest',base_url=base_url)
        chain = prompt | model | parser
        
        payload = chain.invoke({'message':state['message']})
        
        while payload['project_name'] not in projects_info.keys():
            val = input(ask_user(state['message'],f'please provide the project from {projects_info.keys()} only'))
            if val:
                payload['project_name'] = val
            
        
        selected_project_id = projects_info[payload['project_name']]
        payload['project_id'] = selected_project_id
        
        url  ="http://172.16.10.13:5000/add-tasks"
        
        for key in dict(payload).keys():
            if payload[key]=="NO":
                payload[key] = input(ask_user(state['message'],f"Hey! provide task {key}"))
        
        
        response = requests.post(url,json=payload,headers=headers)
        response =response.json()
        
        return{
            'message':state['message'],
            'output':response['message'],
            'user_id':state['user_id']
        }
        
        
        
    except Exception as e:
        return{
            'message':state['message'],
            'output':f"exception {e}",
            'user_id':state['user_id']
        }
        

#************************************** Timesheet module start ******************************************

def create_time_sheet(state: AgentState):
    prompt = PromptTemplate(
        template="""
            Extract the date mentioned in the user's message and return it in a valid JSON format. Follow these rules strictly:

            - If the message refers to **"today"**, replace it with the current date: {date}.
            - If it refers to a **relative date** (e.g., "yesterday", "last Monday", "two days ago"), compute the exact date based on today: {date}, and return that.
            - If the message contains an **explicit date** in `YYYY-MM-DD` or `YYYY/MM/DD` format, extract and return it as is.
            - If **no date is mentioned**, return: {{"date": "DATE_NOT_GIVEN"}}

            Output format:
            - Respond with a valid JSON string only.
            - Do **not** include any explanation, text, markdown, or symbols.

            User message: {message}


        """
        ,
        input_variables=["message","date"]
    )

    model = ChatOllama(model='llama3.3:latest',base_url=base_url)
    parser = JsonOutputParser()
    
    chain = prompt | model | parser
    date= str(datetime.datetime.now()).split(" ")[0]
    print(date)
    
    res = chain.invoke({'message':state['message'],'date':date})
   
    url = "http://172.16.10.13:5000/addtimesheets"
    print(res)
    if res['date']!="DATE_NOT_GIVEN":
        
        response = requests.post(url,json={"start_date": res['date']},headers=headers)
    else:
        date = input(ask_user(state['message'],"please provide the starting date for creating timesheet in yyyy-mm-dd format"))
        response = requests.post(url,json={"start_date": date},headers=headers)
        
    response = response.json()
    print(response)
    if response and response['message'] == "Record added successfully":
        time_sheet = input(ask_user('timesheet created',"do you want to fill the timesheet now?"))
        if 'yes' in  time_sheet.lower():
            url = "http://172.16.10.13:5000/get-tasks-by-org"
            tasks = requests.get(url,headers=headers)
            tasks = tasks.json()
            if tasks['message'] == "No clients found for this organization":
                return {
                    **state,
                    state['output'] : "No clients found for this organization"
                }
            tasks = tasks.json()
            task_info = dict()
            print(tasks)
            
            for i in range(len(tasks['tasks'])):
                task_info[tasks['tasks'][i]['name']] = tasks['tasks'][i]['id']
                
            task = input(ask_user('want to fill the timesheet',f"For which do u want to create the timesheet {task_info.keys()} : "))
            task_id = task_info[task]
            url = "http://172.16.10.13:5000/addtimeentries"
            timesheet_id = response['data']['id']
            hr = float(input(ask_user('want to fill the timesheet',"please provide the number of hours  worked")))
            date = input(ask_user('want to fill the timesheet',"please provide the  date for timesheet in yyyy-mm-dd format"))
            res = requests.post(url=url,json={'task_id':task_id,"timesheet_id":timesheet_id,"hours":hr,'date':date},headers=headers)
                        
            return {
            "message":state['message'],
            "output":res.json()['message'],
            'user_id':state['user_id']
            }
        
    return {
        "message":state['message'],
        "output":response['message'],
        'user_id':state['user_id']
    }
    
def get_timesheet_by_date(date):
    try:
        print('timesheet is fetching.............')
        url = "http://172.16.10.13:5000/get-all-timesheets"
        timesheets = requests.get(url=url,headers=headers)
        timesheets = timesheets.json()
        
        if not timesheets:
            pass
        
        timsheet_info = dict()
        for i in range(len(timesheets['timesheets'])):
            timsheet_info[timesheets['timesheets'][i]['name']]  = timesheets['timesheets'][i]['id']
        
        print('fetched timesheet',timsheet_info)
            
        prompt = PromptTemplate(
            template="""
            You have to select that date range from {range} in which {date} lies.
            - If the date mentioned is **today**, replace it with the actual date value {date}.
            - If the date is relative (e.g., "yesterday", "two days ago", etc.), calculate the correct date based on today's date {date} and use that.
            - Respond in Json format only. Do not include text or explanation or markdown text or symbol

            - Example {{2025-06-11 : 2025-06-08 to 2025-06-15 - akp }} response will be like {{'date':2025-06-11,'range':2025-06-08 to 2025-06-15 - akp}}
            - If date is not in the range then just filled date field with 'NO'
            
            """
        )
        parser = JsonOutputParser()
        model = ChatOllama(model='llama3.3:latest',base_url=base_url)
        chain = prompt | model |parser
        date_res = chain.invoke({'range':timsheet_info.keys(),'date':date})
        print('date is',date_res)
        if date_res['date']=='NO':
            date = input(ask_user('want to fill the timesheet',"please provide the  date for timesheet in yyyy-mm-dd format"))
        timesheet = date_res['range']
        print(timsheet_info,date_res)
        timesheet_id = timsheet_info[timesheet]
        return timesheet_id
        
    except Exception as e:
        raise e
        


def fill_timesheet_entry(state: AgentState):
    try:
        url = "http://172.16.10.13:5000/get-tasks-by-org"
        tasks = requests.get(url,headers=headers)
        tasks = tasks.json()
        if tasks['message'] == 'No projects found for this organization':
            return{
                "message":state['message'],
                "user_id":state['user_id'],
                'output' : ask_user(state['message'],'No projects found for your organization,first create the project')
            }
        task_info = dict()
        
        for i in range(len(tasks['tasks'])):
            task_info[tasks['tasks'][i]['name']] = tasks['tasks'][i]['id']
        
        prompt = PromptTemplate(
            template="""
                You have to extract the **task**, **date**, and **hours** from the user message: {message}.

                - If the date mentioned is **today**, replace it with the actual date value {date}.
                - If the date is relative (e.g., "yesterday", "two days ago", etc.), calculate the correct date based on today's date {date} and use that.
                ***STRICTLY FOLLOW ****
                - If any of the fields (task, hours, or date) are not mentioned in the message, set their value to `'NO'`.

                Return the output strictly in **JSON format** like this:
                {{'task_name': <extracted_task_name>, 'hours': <extracted_worked_hours_float>, 'date': <extracted_date_string>}}

               Return the result as **pure JSON** only — no markdown, no explanation, no extra characters.  

            """,
            input_variables=['message','date']
        )
        
        model_entry = ChatOllama(model='llama3.3:latest',base_url=base_url)
        parser_entry = JsonOutputParser()
        
        chain = prompt | model_entry | parser_entry
        
        d= str(datetime.datetime.now()).split(" ")[0]
        
        res = chain.invoke({'message':state['message'],'date':d})
        print('res is ',res)
        for key in dict(res).keys():
            if res[key] == 'NO':
                res[key] = input(ask_user(state["message"],f"please provide the {key} value"))
                
        while res['task_name'] not in dict(task_info).keys():
            res['task_name'] = input(ask_user(state['message'],f"The task {res['task_name']} is not in the list {dict(task_info).keys()}. Please provide valid task"))
                
        task_id = task_info[res['task_name']]
        timesheet_id = get_timesheet_by_date(res['date'])
        payload = {'timesheet_id':timesheet_id,'task_id':task_id,'hours':res['hours'],'date':res['date']}
        url = "http://172.16.10.13:5000/addtimeentries"
        res = requests.post(url=url,json=payload,headers=headers)
        print("timesheet",res.json())
        
        return {
        "message":state['message'],
        "output":res.json()['message'],
        'user_id':state['user_id']
        }
        
    except Exception as e:
        return{
            "message":state['message'],
        "output":f'exception in timesheet_entry {e}',
        'user_id':state['user_id']
            
        }
        
# ******************************************* Performance report ********************************************************
def performance(state):
    prompt = PromptTemplate(
        template="""
        Your tak is to summerize the performance of employee and generate a report on it based on user question.
        performance_content : {performance}
        question : {question}
        """,
        input_variables=['question','performance']
    )
    
    llm = ChatOllama(model='llama3.3:latest',base_url=base_url)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    memory = load_the_memory(state)
    res = chain.invoke({'performance':memory,'question':state['message']})
    print(res)
    return state


# ***************************** Routing module start *****************************************************************
def router(state: AgentState) -> str:
    """Router logic based on intent"""
    print("Router received:", state)
    if state['intent'] == "register_org":
        return "extract_information_for_register"
    elif state["intent"] == "RAG QUERY":
        return "rag_the_query"
    elif state["intent"]=="normal_conversation":
        return "normal_conversation"
    elif state["intent"]=="login":
        return "login"
    elif state["output"]=="exit":
        return "exit"
    elif state["intent"]=="fall_back":
        return "fall_back"
    elif state['intent'] == "timesheet_creation":
        return 'timesheet_creation'
    elif state['intent'] == "create_client":
        return 'create_client'
    elif state['intent'] == 'create_project':
        return 'create_project'
    elif state['intent'] == 'create_task':
        return 'create_task'
    elif state['intent'] == 'fill_timesheet':
        return 'fill_timesheet'
    elif state['intent'] == 'performance':
        return 'performance'
    else:
        return None
    
def router_for_review(state: str):
    print("router_for_review recived ",state)
    if state['output'] =="payload can not be extracted":
        return "N"
    elif state['output'] =="open_reg_link":
        return "open_reg_link"
    
    elif state['payload']:
        return 'review'
    else:
        return 'N'
# ******************************************************************************* Routing module ended **************************************************


# ************************************************************************** RAG module start ************************************************************
def rag_the_query(state: AgentState):
    print("[TOOL] Rag query is called with input",state)
    
    rag = RAG()
    print("rag recivied",state)
    ans = rag.invoke(state["message"][-1].content)
    print("RAG result:", ans['output'])

    return {
        **state,
        'intent':None,
        "output": ans['output'],
        "message" : state["message"],
        "user_id":state["user_id"],
    }
# **************************************************************************** RAG module ended *********************************************************



def exit(input: AgentState):
    return {
        "output":"exit",
        "message":input["message"],
        "user_id":input["user_id"]
    }



def fall_back(input: AgentState):
    return{
        'output':'This is flaaback',
        "message":input['message'],
        "user_id": input['user_id']
    }

# ========================================= GRAPH WORKFLOW  STARTED ==============================================

workflow = StateGraph(AgentState)
# Add nodes

# workflow.add_node("record_audio_untill_stop", record_audio_untill_stop)
workflow.add_node("decide_the_intent_of_query", decide_the_intent_of_query)
workflow.add_node("ask_for_manual_reg", ask_for_manual_reg)
workflow.add_node("extract_information_for_register", extract_information_for_register)
workflow.add_node("register_the_client", register_the_client)
workflow.add_node("manual_registration", manual_registration)
workflow.add_node("rag_the_query", rag_the_query)
workflow.add_node("normal_conversation",normal_conversation)
workflow.add_node("extract_the_login_parameters",extract_the_login_parameters)
workflow.add_node("login_user",login_user)
workflow.add_node("fall_back",fall_back)
workflow.add_node("generate_review",generate_review)
workflow.add_node("create_time_sheet",create_time_sheet)
workflow.add_node("create_client",create_client)
workflow.add_node("create_project",create_project)
workflow.add_node("create_task",create_task)
workflow.add_node("fill_timesheet",fill_timesheet_entry)
workflow.add_node("performance",performance)
# workflow.add_node("play_audio",play_audio)

# Define entry point
workflow.add_edge("__start__", "decide_the_intent_of_query")
# workflow.add_edge("record_audio_untill_stop", "decide_the_intent_of_query")
# Conditional routing
workflow.add_conditional_edges(
    "decide_the_intent_of_query",
    router,
    {
        "extract_information_for_register": "ask_for_manual_reg",
        "rag_the_query": "rag_the_query",      
        "normal_conversation":"normal_conversation",
        "login":"extract_the_login_parameters",
        "fall_back":'fall_back',
        "timesheet_creation":"create_time_sheet",
        "create_client":"create_client",
        "create_project":"create_project",
        "create_task":"create_task",
        'fill_timesheet':'fill_timesheet',
        "performance":"performance",
        "exit":"__end__"
    }
)

workflow.add_conditional_edges(
    "ask_for_manual_reg",
    lambda state: state["output"],
    {
        "bot_registration": "manual_registration",
        "do you want manual registration? Yes/No ?":"__end__"
         }
)


workflow.add_conditional_edges(
    "manual_registration",
    lambda state: state["output"],
    {
        "bot_registration": "extract_information_for_register",
        "please open the link in browser":"__end__"
         }
)

workflow.add_conditional_edges(
    "extract_information_for_register",
    router_for_review,
    {
        "Y": "__end__",
        "N" :"__end__",
        "review":"register_the_client"
      
    }
)

# Final step: register after extraction
workflow.add_edge("fill_timesheet","__end__")
workflow.add_edge("normal_conversation","__end__")
workflow.add_edge("extract_the_login_parameters","login_user")
workflow.add_edge("register_the_client","__end__")

# Compile the graph
graph_builder= workflow.compile()
# Save graph visualization
with open("graph.png", "wb") as f:
    f.write(graph_builder.get_graph().draw_mermaid_png())                                                                
    

# s = 1
# if s :
#     print("Bot :- hello! I am chronoai an AI assistent for your Timechorono plateform!! How can i help you? ") 
#     s =0

# if __name__ == "__main__":
#     while True:
#         user_input = HumanMessage(input("user :- "))
#         curr_state
#         result = graph_builder.invoke({"message":user_input,"user_id": "1231",})
#         print("Bot :-  ",result["output"])
#         if result["output"] =="exit":
#             break