from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores.chroma import Chroma



 
CHROMA_DB_DIR : str = "./RAG/chroma_db/"
KAGGLE_DATASET_PATH : Path = Path("./1300-towards-datascience-medium-articles-dataset/medium.csv")


class RAG:
    __df : pd.DataFrame = None
    __path : Path 
    __documents : list[Document] = None
    __db : Chroma = None
    __metadata_field_info : list[AttributeInfo] = None
    __document_content_description : str = None
    __llm : HuggingFaceEndpoint = None

    def __init__(self, metadata_field_info : list[AttributeInfo], document_content_description : str,
                  r_splitter : RecursiveCharacterTextSplitter, embeddings : HuggingFaceBgeEmbeddings,llm : HuggingFaceEndpoint = None, ):
        self.__metadata_field_info = metadata_field_info
        self.__document_content_description  = document_content_description
        self.__llm = llm

        self.__load_date(KAGGLE_DATASET_PATH)
        self.__create_documents(r_splitter)
        self.__load_vector_db(embeddings, CHROMA_DB_DIR)


    def __load_date(self, path_to_folder : Path) -> pd.DataFrame:
        df : pd.DataFrame = pd.read_csv(path_to_folder)
        self.__df = df
        self.__path = path_to_folder
        return df


    def __create_documents(self, r_splitter : TextSplitter) -> list[Document]:
        text_list = self.__df['Text'].tolist()
        title_list = self.__df['Title'].tolist()
        self.__documents = r_splitter.create_documents(texts = text_list, metadatas = [{"Title" : title, "from file" : self.__path.name} for title in title_list])
        return self.__documents


    def __load_vector_db(self, embeddings: HuggingFaceBgeEmbeddings, path_to_db :Path) -> Chroma:

        if not os.listdir(path_to_db):
            self.__db = Chroma.from_documents(self.__documents, embeddings,collection_name='db_embeddings' ,persist_directory=path_to_db)
            print(self.__db.similarity_search("what is SQL"))

        else:
            print(os.getcwd())
            print(os.listdir(os.getcwd()))
            print(os.listdir(os.getcwd() + '/' + 'RAG/'))
            print(os.listdir(path_to_db))


            self.__db = Chroma(collection_name='db_embeddings', persist_directory=path_to_db, embedding_function=embeddings)
            print(self.__db.similarity_search("what is SQL"))

        return self.__db



    def find_documents(self, query):        
        return self.__retrive_document(query, self.__llm)


    def __retrive_document(self, query : str, llm : HuggingFaceEndpoint = None ) -> list[Document]:
        answer : list[Document]
        if llm:
            retriever = SelfQueryRetriever.from_llm(
            llm,
            self.__db,
            self.__document_content_description,
            self.__metadata_field_info,
            enable_limit=True,
            verbose=True
        )
            try:
                answer = retriever.invoke(query)
            except ValueError:
                answer = self.__db.max_marginal_relevance_search(query, fetch_k=4, k = 2)
        else:
            answer = self.__db.max_marginal_relevance_search(query, fetch_k=4, k = 2)

        ret_ans : str = ''
        for doc in answer:
            ret_ans += doc
            ret_ans += '\n'
        return ret_ans




def get_RAG_model() -> RAG:
    return __create_RAG_model()



def __create_RAG_model() -> RAG:

    metadata_field_info : list[AttributeInfo] = [
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
    llm : HuggingFaceEndpoint | None
    if os.getenv('HUGGINGFACE_API_KEY') == None:
        llm = None
    else: 
        llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'))
        llm.temperature = 0.1

    r_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 200,
            separators=['\n\n', '\n', '(?<=\.)',' ', '']

        )


    
    model_name = "BAAI/bge-small-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

    # embeddings : HuggingFaceEmbeddings= HuggingFaceEmbeddings()
    return RAG(metadata_field_info, document_content_description, r_splitter, embeddings, llm)

    


if __name__ == '__main__':
    pass
    