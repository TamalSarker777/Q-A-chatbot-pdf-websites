import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
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



# Huggingface environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.accesstoken






def create_vectorstore(url):

    if url is None:
        return

    try:
        loader = WebBaseLoader(url)
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading website content: {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    vectorstore = FAISS.from_documents(docs, huggingface_embeddings)
    st.session_state.vectorstore = vectorstore
    
    
    
def get_retriever(vectorstore):

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
    
    
def define_prompt():
    prompt_template = """
    Use the following piece of context to answer the question asked.
    Please try to provide the answer only based on the context

    {context}
    Question:{question}

    Helpful Answers:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt
    

def main():
    st.title("Chat with :green[Website..]", anchor=False)
    url = st.sidebar.text_input("Enter website URL")
    if not url:
        st.info("Please input website url..")

    # Create vector store and retriever only when a URL is entered
    if url:
        st.session_state.vectorstore = None
        create_vectorstore(url)
        if st.session_state.get("vectorstore") is not None:
            retriever = get_retriever(st.session_state.vectorstore)
            prompt = define_prompt()
            retrievalQA=RetrievalQA.from_chain_type(
                      llm=st.session_state.llm,
                      chain_type="stuff",
                      retriever=retriever,
                       return_source_documents=True,
                       chain_type_kwargs={"prompt":prompt}
                       )
            st.session_state.retrievalqa = retrievalQA
            
        query = st.text_input("Ask a question about the Website")

        if query:
            # Call the QA chain with our query.
            with st.spinner("Generating Answers.."):
                result = st.session_state.retrievalqa.invoke({"query": query})
            with st.chat_message("assistant"):
                st.markdown(result['result'])
            
  
try:
    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None


    # Import Model
    llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03    
        )
        
    if "llm" not in st.session_state:
            st.session_state.llm = llm 

    #define the function
    main()
    
except:
    st.info("Enter your huggingface access token first..")
    
