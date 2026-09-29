from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models.Ling_llm import llm



loader = DirectoryLoader(path='./data', glob='*.pdf', loader_cls=PyPDFLoader)
embeddingModel = HuggingFaceEmbeddings()

raw_docs = loader.load()

text_splittter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=75)

docs=text_splittter.split_documents(raw_docs)


vector_store = Chroma.from_documents(documents=docs, embedding=embeddingModel,
                                        )

retriever = vector_store.as_retriever()

prompt=PromptTemplate(template='my query : {query} and this is retrive doc :{r_docs} ',input_variables=["query","r_docs"])
parser=StrOutputParser()

chain= prompt | llm | parser


while(query:=input('You : ') ):
    if query =='exit':
        break
    result = retriever.invoke(query)

    print(chain.invoke({'query':query,"r_docs":result[:2]}))

    for idx, doc in enumerate(result[:2]):
        print(f"--- Document {idx+1} ---")
        print(doc.page_content)


