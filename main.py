from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders.text import TextLoader
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()


#add here the downloading form kaggle
list_documents = []
path_to_folder : Path = Path("../archive/medium.csv")

# Read the CSV file into a DataFrame
df = pd.read_csv(path_to_folder)
df = df.head(10)

text_list = df['Text'].tolist()
title_list = df['Title'].tolist()

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
    separators=['\n\n', '\n', '(?<=\.)',' ', '']

)

documents = r_splitter.create_documents(texts = text_list, metadatas = [{"Title" : title, "from file" : path_to_folder.name} for title in title_list])


embeddings = HuggingFaceEmbeddings()



from langchain.vectorstores.chroma import Chroma
db_dir_name = "./chroma_db"

print(1)
if not os.path.exists(db_dir_name):
    print(2)
    db = Chroma.from_documents(documents, embeddings, persist_directory=db_dir_name)
else:
    print(3)
    db = Chroma(persist_directory=db_dir_name, embedding_function=embeddings)

print(4)
query = "What is SQL?"

# dosc = db.max_marginal_relevance_search(query, fetch_k = 4,k = 2)
# print(dosc)
print('\n\n\n', '=' * 100, '\n\n\n')

from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


hft = os.getenv('HUGGINGFACE_API_TOKEN')
# os.environ['HUGGINGFACE_API_TOKEN'] = hft
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token=hft)

metadata_field_info = [
    AttributeInfo(
        name="from file",
        description="source of the article",
        type="string",
    ),
    AttributeInfo(
        name="Title",
        description="The title of the article that text is about",
        type="string",
    ),
]


document_content_description = "Chunks of computer science articles"
retriever = SelfQueryRetriever.from_llm(
    llm,
    db,
    document_content_description,
    metadata_field_info,
    verbose=True
)

answer = retriever.get_relevant_documents(query)

# think if i dont want to later use dirrent one to always get 2 best for example


print(answer)

