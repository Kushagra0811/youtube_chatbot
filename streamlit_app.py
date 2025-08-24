from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)   

from urllib.parse import urlparse, parse_qs
import streamlit as st

os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets['HUGGINGFACEHUB_API_TOKEN']

def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from YouTube URLs (works for both long and short formats).
    """
    parsed_url = urlparse(url)

    # For long format: https://www.youtube.com/watch?v=VIDEO_ID
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    # For short format: https://youtu.be/VIDEO_ID
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None
    


load_dotenv()


# --- Functions from your original script ---
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


# --- Streamlit App ---
st.title("YouTube Chatbot")

# Input video ID
raw_url = st.text_input("Enter YouTube URL:")
video_id = extract_video_id(raw_url)


# Main function to run the chatbot
def run_chatbot(video_id, question):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])

        full_transcript = ""
        for chunk in transcript_list:
            full_transcript += chunk.text + " "

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_transcript])

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt = PromptTemplate(
        template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript content. 
            If the context is insufficient, just say you don't know.
            Always try to give a detailed response
            {context}
            Question: {question}
        """,
        input_variables=["context", "question"],
    )

    parser = StrOutputParser()

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    final_chain = parallel_chain | prompt | llm | parser

    result = final_chain.invoke(question)
    return result


# Input question
question = st.text_input("Ask a question about the video:")

# Run button
if st.button("Get Answer"):
    if video_id:
        answer = run_chatbot(video_id, question)
        if answer:
            st.write("Answer:", answer)
    else:
        st.warning("Please enter a YouTube Video ID.")
