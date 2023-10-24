import streamlit as st
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS,Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmltemplates import css, bot_template, user_template



def get_conversation_chain(vectorstore):
   llm=ChatOpenAI()
   #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
   memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
   conversaation_chain=ConversationalRetrievalChain.from_llm(
       llm=llm,
       retriever=vectorstore.as_retriever(),
       memory=memory
       )
   return conversaation_chain


def get_vectorstore(text_chunks):
       embeddings=OpenAIEmbeddings()
       #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
       vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
       return vectorstore



def get_raw_chunk(doc_text):
    text_splitter=CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks=text_splitter.split_text(doc_text)
    return chunks


def get_raw_text(docs_files):
    text=''
    for doc in docs_files:
        pdf_reader=PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_web(web_link):
    import requests
    from bs4 import BeautifulSoup
    url =web_link
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
          # Parse the HTML content using BeautifulSoup
          soup = BeautifulSoup(response.content, 'html.parser')
          # Find all the text within the HTML and join it together
          text = ' '.join([p.get_text() for p in soup.find_all('p')])
          return text
    else:
         print("Failed to fetch the web page")



def handle_userinput(user_questions):
    response = st.session_state.conversation({'question': user_questions})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    

def main():
    load_dotenv()
    st.set_page_config(page_title='smart Chatbot based on your documents',page_icon='https://icons.iconarchive.com/icons/graphicloads/colorful-long-shadow/48/Files-icon.png',layout='wide')
    st.write(css, unsafe_allow_html=True)
    st.header("Chat With Your Documents:book:")
    user_questions=st.text_input("Ask a question about your documents:")
    if user_questions:
        handle_userinput(user_questions)
    with st.sidebar:
        st.subheader("Your documents".title())
        docs_files = st.file_uploader("Upload your Documents here and click on 'Process'", accept_multiple_files=True)
        st.subheader("If you have any questions about any website".upper())
        web_link=st.text_input("please upload the link here ! else skip it ")
        if st.button("Process"):
            with st.spinner("Processing ..."):
                if web_link:
                    raw_text=get_text_web(web_link)
                    #with  st.success('DONE ...'):
                    # convert pdf into  text
                raw_text=get_raw_text(docs_files)
                # convert text into chunks
                text_chunks=get_raw_chunk(raw_text)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__=="__main__":
      main()