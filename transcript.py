from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# Initialize the language model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0.3, max_output_tokens=1024, top_p=1, top_k=1
)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=10000, chunk_overlap=1000, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


def get_conversation_chain():
    prompt_template = """
    You are an expert assistant in summarizing YouTube transcripts. Please provide a concise summary of the following transcript:

    {text}

    Summary:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain


# Step 1: Train the model using a .txt file
def summarize_text(file_path):
    with open(file_path, "r") as f:
        transcript = f.read()

    text_chunks = get_text_chunks(transcript)
    vectorstore = get_vectorstore(text_chunks)
    chain = get_conversation_chain()

    docs = [Document(page_content=chunk) for chunk in text_chunks]
    summary = chain.run(docs)

    return summary, vectorstore


file_path = "transcript.txt"
summary = summarize_text(file_path)
print(summary)


def get_vectorstore_1(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore_1


# Step 2: Fine-tune the model with the JSON file
def fine_tune_model_with_json(vectorstore_1, json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    text_chunks = [item["text"] for item in data]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_1.add_texts(text_chunks, embeddings=embeddings)

    # Save the fine-tuned model
    vectorstore_1.save_local("faiss_index_fine_tuned")
    return vectorstore_1


# Step 3: Summarize a specific duration from JSON data
def summarize_text_by_duration(json_file_path, start_time, end_time):

    text_chunks = load_json_data_by_duration(json_file_path, start_time, end_time)

    chain = get_conversation_chain()

    docs = [Document(page_content=chunk) for chunk in text_chunks]

    summary_j = chain.run(docs)
    return summary_j


# Function to load and extract text from the JSON file based on start and end times
def load_json_data_by_duration(json_file_path, start_time, end_time):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Filter the data based on start_time and end_time
    filtered_texts = [
        item["text"] for item in data if start_time <= item["offset"] <= end_time
    ]

    return filtered_texts


txt_file_path = " .txt"  # Path to your .txt file
json_file_path = ".json"  # Path to your JSON file

# Train model using the .txt file
summary_j, vectorstore = summarize_text(txt_file_path)

# Fine-tune model using the JSON file
vectorstore = fine_tune_model_with_json(vectorstore, json_file_path)

# Summarize text between specific times from the JSON file
start_time = 1280  # Example start time in milliseconds
end_time = 9639  # Example end time in milliseconds
summary_j = summarize_text_by_duration(json_file_path, start_time, end_time)
print(summary_j)
