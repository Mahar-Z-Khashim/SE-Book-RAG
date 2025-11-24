#Libraries
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate


#Change user and assistant emojis colors 
st.markdown(
    """
    <style>
    .st-emotion-cache-23r7bk {
        background-color: #f7b63b !important;  /* user */
    }
    
    .st-emotion-cache-1niilgc {
        background-color: #EA028C !important;  /* assistant */
    }
    </style>
    """,
    unsafe_allow_html=True
)


#Generate the database if it's not exist
if not os.path.exists("chroma_SE_RAG"):
    import build_db  


#Load openai key
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("✅ OPENAI_API_KEY loaded.")
else:
    print("❌ OPENAI_API_KEY is missing. Please set it in a .env file.")


#Load the vector database and store it in cache to make the system faster
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory="chroma_SE_RAG",
        embedding_function=embeddings
    )

vectordb = load_vectordb()

retriever = vectordb.as_retriever(search_kwargs={"k": 4})


#Prompt
prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering.

Use ONLY the following context to answer.

Context:
{context}

Question:
{question}

Answer concisely and clearly.
""")


#RAG + LLM
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
)

print("✅ RAG chain ready.")


#Streamlit
st.title("📘 SE RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

#Show previous messages
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

#User input
if question := st.chat_input("Ask about SE Book..."):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.history.append(("user", question))

    #Show spinner while the RAG chain is processing
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking... please wait"):
            response = rag_chain.invoke(question)
        st.markdown(response.content)

    st.session_state.history.append(("assistant", response.content))
