import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import os


st.sidebar.image('mypic.png', width=25)

if "accesstoken" not in st.session_state:
    st.session_state.accesstoken = None
    
def access_key(access_token):
    st.session_state.accesstoken = access_token

access_token = st.sidebar.text_input("Enter your huggingface access token..", key=str, placeholder="Enter Key..")

st.sidebar.button("Enter", on_click=access_key(access_token))





# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None



def find_vectorstore(data):
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs=text_splitter.split_documents(data)
    
    huggingface_embeddings=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    vectorstore=FAISS.from_documents(docs,huggingface_embeddings)
    
    return vectorstore





def upload():
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner('Getting Ready, Please be patient...'):
            # Load and process the PDF file
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
        
            # Store the extracted data in session state
            st.session_state.data = data
            
            
            st.session_state.vectorstore = find_vectorstore(data)
      
        os.remove("temp.pdf")






#Huggingface environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.accesstoken

try:
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.accesstoken
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03    
    )
    
    if "llm" not in st.session_state:
        st.session_state.llm = llm 
        
    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")


    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        
        
    # Button to trigger the upload function
    st.sidebar.button(":blue[Upload]", on_click=upload)

    # Main app title and display the data
    st.title("Chat With :green[PDF..]", anchor=False)
    
    
    
    
    try:
        vectorstore = st.session_state.vectorstore
        retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})


        prompt_template="""
        Use the following piece of context to answer the question asked.
        Please try to provide the answer only based on the context

        {context}
       Question:{question}
       
        Helpful Answers:
        """

        prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])



        retrievalQA=RetrievalQA.from_chain_type(
                   llm=st.session_state.llm,
                   chain_type="stuff",
                   retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt":prompt}
                    )
        
        if "retrievalqa" not in st.session_state:
            st.session_state.retrievalqa = retrievalQA
            
        
        
        
        
        query = st.text_input("Ask a question about the PDF file")
        if query:
            # Call the QA chain with our query.
            retrievalQA = st.session_state.retrievalqa
            with st.spinner("Generating Answers.."):
                result = retrievalQA.invoke({"query": query})
            with st.chat_message("assistant"):
                st.markdown(result['result'])
                
    except:
        st.info("Please Upload first.....")

except:
    st.info("Enter your Huggingface access key first..")





    

#import Model
# llm = Ollama(model="llama2")

# if "llm" not in st.session_state:
#     st.session_state.llm = llm  



    
    
    



















