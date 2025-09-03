from langchain_community.vectorstores import Chroma
from generate_dataset import load_documents_from_data_folder
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
llm = ChatOpenAI(
    model="qwen-flash",
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)
embeddings = DashScopeEmbeddings( model="text-embedding-v4", dashscope_api_key=os.environ.get("OPENAI_API_KEY"))
# embeddings = OpenAIEmbeddings(model="text-embedding-v4",
#                             api_key=os.environ.get("OPENAI_API_KEY"), 
#                             base_url=os.environ.get("OPENAI_API_BASE"))
if __name__ == "__main__":
    shipping_docs = load_documents_from_data_folder("data")
    shipping_db_directory = 'db/shipping/'
    if not os.path.exists(shipping_db_directory):
        vectordb = Chroma.from_texts(
            metadatas=[{"index":i.split("_")[0]} for i in os.listdir("data")],
            collection_name="shipping",
            texts=shipping_docs,
            embedding=embeddings,
            persist_directory=shipping_db_directory
        )