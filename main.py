from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from pathlib import Path
import openai
import os


path_to_folder : Path = Path("../archive/medium.csv")
# test if langchain will work



# 1. find the way of storing the articles in the documents

loader : CSVLoader = CSVLoader(path_to_folder)
pages : list[Document]= loader.load_and_split()

print(len(pages))
# for page in pages[:5]:
#     print(page, end= '\n\n')
# print(pages[0])

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150,
    separators=['\n\n', '\n', ' ', '']

)

splitted = r_splitter.split_documents(pages)

print(len(splitted))
# for spl in splitted:
#     print('\n\n', spl)


# create embedings
# embeddings = OpenAIEmbeddings() 
embeddings = HuggingFaceEmbeddings()

index_current = "curent_index"

print(1)
if not os.path.exists(index_current):
    print(2)

    db = FAISS.from_documents(splitted, embeddings)
    print(3)
    db.save_local(folder_path=index_current)
else:
    db = FAISS.load_local(index_current, embeddings, allow_dangerous_deserialization=True)

# sentence = "I like dogs"
# print(embeddings.embed_query(sentence))

print(4)
query = "What is gensim library and how to install it"

dosc = db.similarity_search(query)
print(dosc)
