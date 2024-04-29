from pathlib import Path
from dotenv import load_dotenv
import os

import pandas as pd
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores.chroma import Chroma
import opendatasets as od

load_dotenv(dotenv_path='config.env')

path_to_folder : Path = Path("./1300-towards-datascience-medium-articles-dataset/medium.csv")

def load_dataset():
    if not os.path.exists(path_to_folder):
        od.download("https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset/data","./")


class RAG:
    __df : pd.DataFrame = None
    __documents : list[Document] = None
    __db : Chroma = None
    __metadata_field_info : list[AttributeInfo] = None
    __document_content_description : str = None

    def __init__(self, metadata_field_info : list[AttributeInfo], document_content_description : str):
        self.__metadata_field_info = metadata_field_info
        self.__document_content_description  = document_content_description


    def load_date(self, path_to_folder : Path) -> pd.DataFrame:
        df = pd.read_csv(path_to_folder)
        df = df.head(10)
        self.__df = df
        return df


    def create_documents(self, r_splitter : TextSplitter) -> list[Document]:
        if not (self.__df and self.__documents):
            return
        text_list = self.__df['Text'].tolist()
        title_list = self.__df['Title'].tolist()
        self._documents = r_splitter.create_documents(texts = text_list, metadatas = [{"Title" : title, "from file" : path_to_folder.name} for title in title_list])
        return self.__documents


    def load_vector_db(self, embeddings: HuggingFaceEmbeddings, path_to_db :Path) -> Chroma:

        if not os.path.exists(path_to_db):
            self.__db = Chroma.from_documents(self.__documents, embeddings, persist_directory=path_to_db)
        else:
            self.__db = Chroma(persist_directory=path_to_db, embedding_function=embeddings)

        return self.__db



    def find_documents(self):
        pass

    def retrive_document(self, llm : HuggingFaceEndpoint, query : str) -> list[Document]:
        retriever = SelfQueryRetriever.from_llm(
        llm,
        self.__db,
        self.__document_content_description,
        self.__metadata_field_info,
        verbose=True
    )
        try:
            answer = retriever.invoke(query)
        except ValueError:
            print('Error')
            answer = self.__db.max_marginal_relevance_search(query, fetch_k=4, k = 2)
        
        return answer







if __name__ == '__main__':

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
    llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'))

    r_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 200,
            separators=['\n\n', '\n', '(?<=\.)',' ', '']

        )

    embeddings = HuggingFaceEmbeddings()
    db_dir_name = "./chroma_db"

    query = "What is SQL?"

    