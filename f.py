import os
from apikey import apikey
from config import sysprompt, datafile
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st



os.environ["OPENAI_API_KEY"] = apikey
llm = ChatOpenAI ()
embeddings = OpenAIEmbeddings ()


mem = ConversationBufferMemory (
    chat_memory = FileChatMessageHistory ("mem/messages.json"),
    memory_key = "chat_history",
    return_messages = True
)

# chain = ConversationalRetrievalChain.from_llm (llm = chat, memory = memory, retriever = retriever)

def embedWeb (url, location):
    from langchain.document_loaders import WebBaseLoader
    loader = WebBaseLoader (url)

    data = loader.load ()
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents (data)
    db = Chroma.from_documents (texts, embedding = embeddings, persist_directory = location)

    st.write ("Done embedding")


def embedYoutube (url, location):
    from langchain.document_loaders import YoutubeLoader
    loader = YoutubeLoader.from_youtube_url (url)

    data = loader.load ()
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents (data)
    db = Chroma.from_documents (texts, embedding = embeddings, persist_directory = location)

    st.write ("Done embedding")


def embedDoc (file, location):
    name, extension = os.path.splitext (file)
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader (file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader (file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader (file)
    else:
        st.write ('Document format is not supported')

    data = loader.load ()
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents (data)
    db = Chroma.from_documents (texts, embedding = embeddings, persist_directory = location)




def queryData (location):
    db = Chroma (embedding_function = embeddings, persist_directory = location)
    retr = db.as_retriever ()
    # chat_history = []

    retrPrompt = ChatPromptTemplate.from_messages ([
        MessagesPlaceholder (variable_name = "chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    retrChain = create_history_aware_retriever (llm, retr, retrPrompt)

    docPrompt = ChatPromptTemplate.from_messages ([
        # ("system", "Answer the user's questions based on the below context:\n\n {context}"),
        # ("system", "Answer only from retrieved document and nowhere else. If you do not know the answer, just say I don't know:\n\n {context}"),
        # ("system", "You are a joker. Always answer with a joke:\n\n {context}"),
        ("system", sysprompt),
        MessagesPlaceholder (variable_name = "chat_history"),
        ("user", "{input}")
    ])
    docChain = create_stuff_documents_chain (llm, docPrompt)


    # chat_history = [
    #     HumanMessage (content = "Who is Leonardo Vetra?"),
    #     AIMessage (content = "He's a scientist at CERN.")
    # ]
    rChain = create_retrieval_chain (retrChain, docChain)
    question = st.text_input ("Input your question")
    if question:
        if 'history' not in st.session_state: st.session_state['history'] = []

        r = rChain.invoke ({"chat_history": st.session_state['history'], "input": question})
        st.session_state['history'].append (HumanMessage(question))
        st.session_state['history'].append (AIMessage(r['answer']))
        st.write (r['answer'])
        print (st.session_state['history'])


st.title ('Chat with Data')
# embedDoc ("data/book.pdf", "emb/danbrown")
# embedWeb ("https://nyp.edu.sg", "emb/nypweb")
queryData (datafile)
