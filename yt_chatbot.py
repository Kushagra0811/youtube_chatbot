from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
load_dotenv()
# Step 1 indexing

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text





video_id = "N0_9Q-G2KL4"  # Only the ID, not the full URL
transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['hi'])

try:
    transcript = ""
    for chunk in transcript_list:
        transcript += chunk.text + " "
    

except TranscriptsDisabled:
    print("No captions available for this video")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)


retriever = vector_store.as_retriever(search_type = 'similarity',search_kwargs = {'k':4})

result = retriever.invoke('What is deepmind')


llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature = 0.2)

prompt = PromptTemplate(
    template = """
        You are a helpful assistant.
        Answer ONLY from the provided transcript content. 
        If the context is insufficient, just say you don't know.

        {context}
        Question : {question}
    """,
    input_variables=['context','question']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
})

final_chain = parallel_chain | prompt | llm | parser
question = 'Was samay able to code, and what interesting things happened'
result = final_chain.invoke(question)
print(result)

