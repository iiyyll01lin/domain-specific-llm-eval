import streamlit as st
import taskingai
import os
from taskingai.assistant import *
from taskingai.assistant.memory import AssistantNaiveMemory
from streamlit_option_menu import option_menu


TASKINGAI_SERVICE="http://ai-platform-uat.eastus.cloudapp.azure.com:8080"
TASKINGAI_API="tkIVY7ippvfzemSZUbZ3sK8VS5DUDYsL"
TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT35_ID="X5lMb4OLX2zuQoE835TnSb2h"
TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4_ID="X5lMKmYm4Kq3FXbAvojOSXH0"
TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4O_ID="X5lMyquYoiwIzD3sNi7lN47u"
TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT35_ID="X5lMK9fHIXIuLG3MVTBF0Hmf"
TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT4O_ID="X5lMMqLyjWtGiVfCP37Lb2Jh"
TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT35_ID="X5lMBAGIJtxbfWfA3SqeyMUF"
TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT4O_ID="X5lMjKW6vkoUrZd3mf2AmMMW"


rv = taskingai.init(api_key=TASKINGAI_API, host=TASKINGAI_SERVICE)

if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = ""

if "assistant_id_previous" not in st.session_state:
    st.session_state.assistant_id_previous = ""

if "chat_id" not in st.session_state:
    st.session_state.chat_id =  {
                                 TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT35_ID: "",
                                 TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4_ID: "",
                                 TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4O_ID: "",
                                 TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT35_ID: "",
                                 TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT4O_ID: "",
                                 TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT35_ID: "",
                                 TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT4O_ID: "",
                                }
    chat_session = taskingai.assistant.create_chat(
        assistant_id=TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT35_ID,
    )
    st.session_state.chat_id[TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT35_ID] = chat_session.chat_id

    chat_session = taskingai.assistant.create_chat(
        assistant_id=TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4_ID,
    )
    st.session_state.chat_id[TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4_ID] = chat_session.chat_id

    chat_session = taskingai.assistant.create_chat(
        assistant_id=TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4O_ID,
    )
    st.session_state.chat_id[TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4O_ID] = chat_session.chat_id
    
    chat_session = taskingai.assistant.create_chat(
        assistant_id=TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT35_ID,
    )
    st.session_state.chat_id[TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT35_ID] = chat_session.chat_id

    chat_session = taskingai.assistant.create_chat(
        assistant_id=TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT4O_ID,
    )
    st.session_state.chat_id[TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT4O_ID] = chat_session.chat_id
    
    chat_session = taskingai.assistant.create_chat(
        assistant_id=TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT35_ID,
    )
    st.session_state.chat_id[TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT35_ID] = chat_session.chat_id

    chat_session = taskingai.assistant.create_chat(
        assistant_id=TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT4O_ID,
    )
    st.session_state.chat_id[TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT4O_ID] = chat_session.chat_id

if "messages" not in st.session_state:
    st.session_state.messages = {
                                 TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT35_ID: [],
                                 TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4_ID: [],
                                 TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4O_ID: [],
                                 TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT35_ID: [],
                                 TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT4O_ID: [],
                                 TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT35_ID: [],
                                 TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT4O_ID: [],
                                }

    def get_message_list(assistant_id,chat_id,message_id):
        print(assistant_id)
        print(chat_id)
        result = taskingai.assistant.list_messages( 
            assistant_id=assistant_id, 
            chat_id=chat_id, 
            limit=100, 
            order="asc",
            after=message_id)

        return result

    for key, value in st.session_state.chat_id.items():
        history = []
        assistant_id=key
        chat_id=value
        retrieval = True
        last_message_id = None
        while retrieval :
            message_list = get_message_list(assistant_id, chat_id, last_message_id)
            history.extend(message_list)
            if len(message_list) == 100 :
                last_message_id = message_list[99].message_id
            else:
                retrieval = False

        for index in range(len(history)):
            if history[index].role == "user" :
                    st.session_state.messages[assistant_id].append({"role": "user", "content": history[index].content.text})
            else :
                    st.session_state.messages[assistant_id].append({"role": "assistant", "content": history[index].content.text})
        
# Streamlit UI
# ===================================================================================================

st.set_page_config(
    page_title = 'ROMANTIC RUSH AI Chatbot',
    page_icon = '🤖',
    layout = 'wide'
)

st.title("Chat 💬🗨️")

st.sidebar.title("🤖 ROMANTIC RUSH")

st.sidebar.markdown("-------------------------")


with st.sidebar:
    rag_selected = option_menu("Assistant", ["Purchasing", "HR Rules", "Engineering" ],
        icons=['robot', 'robot', 'robot'], menu_icon="gear", default_index=0)

st.sidebar.markdown("-------------------------")

if rag_selected == "Engineering" :
    with st.sidebar:
        model_selected = option_menu("LLM Models", ["MistraAI", 'Gemini-1.5-pro', 'ChatGPT-4o'],
            icons=['cpu-fill', 'cpu-fill', 'cpu-fill'], menu_icon="cpu", default_index=0)
else:
    with st.sidebar:
        model_selected = option_menu("LLM Models", ["Gemini-1.5-pro", 'ChatGPT-4o'], 
            icons=['cpu-fill', 'cpu-fill', 'cpu-fill'], menu_icon="cpu", default_index=0)

if model_selected == "Gemini-1.5-pro" and rag_selected == "Purchasing" :
    st.session_state.assistant_id = TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT35_ID
if model_selected == "ChatGPT-4o" and rag_selected == "Purchasing" :
    st.session_state.assistant_id = TASKINGAI_INVENTEC_FINANCIAL_ASSISTANCE_CHATGPT4O_ID
if model_selected == "Gemini-1.5-pro" and rag_selected == "HR Rules" :
    st.session_state.assistant_id = TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT35_ID
if model_selected == "ChatGPT-4o" and rag_selected == "HR Rules" :
    st.session_state.assistant_id = TASKINGAI_IT_SHIFT_PLAN_ASSISTANCE_CHATGPT4O_ID
if model_selected == "MistraAI" and rag_selected == "Engineering" :
    st.session_state.assistant_id = TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT35_ID
if model_selected == "Gemini-1.5-pro" and rag_selected == "Engineering" :
    st.session_state.assistant_id = TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4_ID
if model_selected == "ChatGPT-4o" and rag_selected == "Engineering" :
    st.session_state.assistant_id = TASKINGAI_INVENTEC_KNOWLEDGEBASE_ASSISTANCE_CHATGPT4O_ID

# Display chat messages from history on app rerun

for message in st.session_state.messages[st.session_state.assistant_id]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("What do you want to ask for the Assistant?"):
    # Display your message
    with st.chat_message("user"):
        st.markdown(user_question)
    # Add your message to chat history
    st.session_state.messages[st.session_state.assistant_id].append({"role": "user", "content": user_question}) 
    
    print("Q:{0}".format(user_question))

    taskingai.assistant.create_message(
        assistant_id=st.session_state.assistant_id,
        chat_id=st.session_state.chat_id[st.session_state.assistant_id],
        text=user_question,
    )

    # generate assistant response

    assistant_message = taskingai.assistant.generate_message(
        assistant_id=st.session_state.assistant_id,
        chat_id=st.session_state.chat_id[st.session_state.assistant_id],
        stream=True,
    )

    with st.spinner():
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            string = ""
            for item in assistant_message:
                if isinstance(item, MessageChunk):
                    message_placeholder.markdown(string + "▌")
                    string=string+"{0}".format(item.delta)
            message_placeholder.markdown(string)

    print("A:{0}".format(string))
    # append the response to chat history
    st.session_state.messages[st.session_state.assistant_id].append({"role": "assistant", "content": string})
    print("================================")
