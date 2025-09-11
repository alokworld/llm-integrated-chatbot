import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_API_KEY = "sample-key"

# Upload the PDF files
st.header("My First Chatbot")
st.title("Your Documents")
file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)

    # Embeddings and storing them into Vector Store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Creating vector - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user questions
    user_question = st.text_input("Type your question here")

    # Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        st.write(match)

        # Define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(question=user_question)
        st.write(response)
      
