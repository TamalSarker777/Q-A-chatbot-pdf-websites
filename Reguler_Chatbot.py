import streamlit as st
# from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
import streamlit.components.v1 as components
from langchain_huggingface import HuggingFaceEndpoint
# import time

import os



st.sidebar.image('mypic.png', width=25)

if "accesstoken" not in st.session_state:
    st.session_state.accesstoken = None
    
def access_key(access_token):
    st.session_state.accesstoken = access_token

access_token = st.sidebar.text_input("Enter your huggingface access token..", key=str, placeholder="Enter Key..")

st.sidebar.button("Enter", on_click=access_key(access_token))

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.accesstoken





model_select = st.sidebar.selectbox(
    ":green[Select A Model From Bellow]",
    ("Mistral-7B", "google/gemma-2b", "zephyr-7b-beta"),
    key="model_selector"
)



slide = st.sidebar.slider(
    ":red[Choose Model Temperature]",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="temperature_selector"
    
    )


Role_play = st.sidebar.selectbox(
    ":green[Select Roleplay Mode]",
    ("Assistant","Drunk", "Emotional", "FBI Agent", "Girlfriend", "Boyfriend"),
    key="play_role"
)

if "role_play" not in st.session_state:
    st.session_state.role_play = None




st.title("Chat With Your :blue[Chatbot] Here :sunglasses: ", anchor=False)

st.warning("Please be aware that, This Q&A chatbot :red[might not] function as expected. It leverages free, open-source models from the Hugging Face pipeline, which may sometimes produce inaccurate responses. In the video demonstration, I downloaded and integrated advanced models such as LLama2, Code Llama, and Mistral-7b locally, which is why it performed flawlessly.\n However, the :red[Chat with PDF] and :green[Chat with Website] features are fully operational and should work seamlessly.\n Thank you for your understanding.")


#--------------------------------------------------
#Generate 4 digit random character
import random
import string
def generate_random_characters(length=4):
    """Generates a random string of the specified length using characters, digits, and punctuation."""
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

#----------------------------------------------------------    




if 'models' not in st.session_state:
    st.session_state.models = None
    

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
    
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Check if history exists for this session ID
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

random_characters = generate_random_characters()





config = {"configurable": {"session_id": f"{random_characters}"}}
if "config" not in st.session_state:
    st.session_state.config = config
    


    
def get_chain():
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are {st.session_state.role_play}. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    )
    output_parser = StrOutputParser()
    chain = prompt| st.session_state.models | output_parser 
    return chain
    
    
    
    
def display_previous_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    



def input_from_user(with_message_history):
    display_previous_chat()
    if input_text:= st.chat_input("Hey!"):
        # Display user message in chat message container
        st.chat_message("user").markdown(input_text)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": input_text})
        
        
        #for LLM response
        response = with_message_history.invoke(
        [HumanMessage(content=input_text)],
        config=st.session_state.config,)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        



   
        

    
    
try:
    #-----------------------------------------Chat Using Mistral-7B Model---------------------------------------------------------------------------

    if st.session_state.model_selector == "Mistral-7B":
        
        mycode = "<script>alert('You select Mistral-7B model')</script>"
        components.html(mycode, height=0, width=0)

        # llm = Ollama(model = "llama2", temperature= st.session_state.temperature_selector)
        # st.session_state.models = llm
        

        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        
        )   
        st.session_state.models = llm

        st.session_state.role_play = st.session_state.play_role
        
        with_message_history = RunnableWithMessageHistory(get_chain(), get_session_history)

            
        input_from_user(with_message_history)


    #---------------------------------------------------------------------------------------------------------------------------






    #-----------------------------------------Chat Using google/gemma-2b Model---------------------------------------------------------------------------

    if st.session_state.model_selector == "google/gemma-2b":
        
        mycode = "<script>alert('You select google/gemma-2b model')</script>"
        components.html(mycode, height=0, width=0)

        # llm2 = Ollama(model = "codellama", temperature= st.session_state.temperature_selector)
        # st.session_state.models = llm2
        
        llm2 = HuggingFaceEndpoint(
            repo_id="google/gemma-2b",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        st.session_state.models = llm2 
        
        
        st.session_state.role_play = st.session_state.play_role
        
        
        with_message_history = RunnableWithMessageHistory(get_chain(), get_session_history)

                
        input_from_user(with_message_history)

    #---------------------------------------------------------------------------------------------------------------------------    
        
        
        
      
        
    #-----------------------------------------Chat Using zephyr-7b-beta Model---------------------------------------------------------------------------

    if st.session_state.model_selector == "zephyr-7b-beta":
        mycode = "<script>alert('You select zephyr-7b-beta model')</script>"
        components.html(mycode, height=0, width=0)
        
        # llm3 = Ollama(model = "mistral", temperature= st.session_state.temperature_selector)
        # st.session_state.models = llm3
        
        
        llm3 = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
         
        st.session_state.models = llm3
        
        st.session_state.role_play = st.session_state.play_role
        
        with_message_history = RunnableWithMessageHistory(get_chain(), get_session_history)
            
        
        input_from_user(with_message_history)


    #---------------------------------------------------------------------------------------------------------------------------
    
except:
    st.info("Enter your huggingface accesskey first..")    
    















