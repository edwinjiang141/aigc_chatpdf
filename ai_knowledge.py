import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from param_config import  ParseConfig,ArgumentParser
from filetype import guess
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from doc_analyze import DocAnalyze
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 


class ai_knowledge:

    def __init__(self,file_inputs: any = None,query: str = None,doc_class: str = None):
        self.file_inputs = file_inputs
        self.query = query
        self.doc_class = doc_class

    def detect_document_type(self,document_path):

        print('document',document_path)
        guess_file = guess(document_path)
        file_type = ""
        image_types = ['jpg', 'jpeg', 'png', 'gif']

        if(guess_file.extension.lower()  == "pdf"):
            file_type = "pdf"

        elif(guess_file.extension.lower() in image_types):
            file_type = "image"

        if(guess_file.extension.lower() == "txt"):
            file_type = "txt"

        else:
            file_type = "unkown"

        return file_type

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_doc_search(self,text_splitter):

        enbeddings = OpenAIEmbeddings(openai_organization = "org-cODSJjftWgVspplR3MwUi2HN",openai_api_key = os.environ["OPENAI_API_KEY"])
        print('text_splitter: %s',text_splitter)
        #return FAISS.from_texts(text_splitter, enbeddings)
        return FAISS.from_documents(text_splitter, enbeddings)

    def extract_file_content(self,file_path):
        file_type = self.detect_document_type(file_path)

        if file_type in ('pdf','unkown'):
            loader = UnstructuredFileLoader(file_path)
        elif file_type == "image":
            loader = UnstructuredImageLoader(file_path)
        
        documents_content = loader.load()
        #documents_content = '\n'.join(doc.page_content for doc in documents)
        # print(type(documents_content))
        return documents_content


    def save_data_fasis(self):

        print('file_path %s',self.file_inputs.name)
        print('doc class %s',self.doc_class)

        file_content = self.extract_file_content(self.file_inputs.name) 
        print('file content %s',file_content)
        text_splitter =  CharacterTextSplitter (
            separator = "\n",
            chunk_size =  3000,  
            chunk_overlap   =  50, 
        )
        text_splitter = text_splitter.split_documents(file_content) 
        docana = DocAnalyze("text-davinci-003")
        print(docana.run(text_splitter))

        document_search = self.get_doc_search(text_splitter)
        if self.doc_class == '故障处理':
            #document_search.save_local("troubleshooting_index")
            try:
                enbeddings = OpenAIEmbeddings(openai_organization = "org-cODSJjftWgVspplR3MwUi2HN",openai_api_key = os.environ["OPENAI_API_KEY"])
                trouble_index = FAISS.load_local("troubleshooting_index", enbeddings)
                trouble_index.merge_from(document_search)
                trouble_index.save_local("troubleshooting_index")
            except Exception as e:
                document_search.save_local("troubleshooting_index")

        elif self.doc_class == 'SQL 优化':
            document_search.save_local("sql_optimizer_index")

    def chat_with_file(self):

        
        enbeddings = OpenAIEmbeddings(openai_organization = "org-cODSJjftWgVspplR3MwUi2HN",openai_api_key = os.environ["OPENAI_API_KEY"])
        #当向量数据库中没有合适答案时，使用大语言模型能力

        if self.doc_class == '故障处理':
            document_search = FAISS.load_local("troubleshooting_index", enbeddings)
        elif self.doc_class == 'SQL 优化':
            document_search = FAISS.load_local("sql_optimizer_index", enbeddings)

        print('docclass %s',self.doc_class)
        print('query %s',self.query)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_organization = "org-cODSJjftWgVspplR3MwUi2HN",openai_api_key = os.environ["OPENAI_API_KEY"])
        qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=document_search.as_retriever(search_type="similarity_score_threshold",
                                                                 search_kwargs={"score_threshold": 0.8}))
        result = qa_chain({"query": self.query})
        return result['result']
        #documents = document_search.similarity_search(self.query)

        #answers = documents
        #print(documents[0].page_content)sss
        # answers = ''

        # for idx,doc in enumerate(documents):
        #     answers += documents[idx].page_content