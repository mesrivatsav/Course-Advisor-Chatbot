import streamlit as st
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def create_vectorstore(value):
    print(value)
    return Chroma(collection_name=value, persist_directory="./chroma_db", embedding_function=OllamaEmbeddings(model='nomic-embed-text'))

def rag_chain(question, vectorstore):
    retriever = vectorstore.as_retriever()

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_prompt(question, context):
        return f"Question: {question}\n\nContext: {context}"

    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    formatted_prompt = format_prompt(question, formatted_context)
    stream = ollama.chat(model=st.session_state["model"], messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    #print(retrieved_docs)
    #print(formatted_context)
    for chunk in stream:
        yield chunk["message"]["content"]

# --- Streamlit App ---
st.set_page_config(layout="wide")  # Set the page layout to wide

st.title("Course Advisory Chatbot")

# Sidebar for filters
with st.sidebar:
    st.header("Filter Options")
    
    # Options for universities
    options = {
        "TUS": "Technological University of the Shannon (TUS)",
        "TUD": "Technological University Dublin (TUD)",
        "SETU": "South East Technological University (SETU)",
        "MTU": "Munster Technological University (MTU)",
        "ATU_SLIGO": "Atlantic Technological University (ATU) - Sligo",
        "ATU_STANGELAS": "Atlantic Technological University (ATU) - St Angelas"
    }

    # University selection
    selected_university = st.selectbox("Select any Technological University", list(options.keys()), format_func=lambda x: options[x])

    # Model selection
    models = [model["name"] for model in ollama.list()["models"]]
    st.session_state["model"] = st.selectbox("Choose your model", models)
    #st.session_state["model"] = "tu-llama2:latest"
    #print(st.session_state["model"])

    # RAG toggle
    st.checkbox("Use RAG", key="use_rag", value=True)

# Main area for the chat interface
if "vectorstore" not in st.session_state or st.session_state.get("selected_university") != selected_university:
    with st.spinner("Loading/Creating vectorstore..."):
        st.session_state["vectorstore"] = create_vectorstore(selected_university)
        st.session_state["selected_university"] = selected_university

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.get("use_rag", True):
            message = st.write_stream(rag_chain(prompt, st.session_state["vectorstore"]))
            st.session_state["messages"].append({"role": "assistant", "content": message})
        else:
            message = st.write_stream(model_res_generator())
            st.session_state["messages"].append({"role": "assistant", "content": message})