from langchain_community.document_loaders import TextLoader,DirectoryLoader,PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()
model=ChatOpenAI()

prompt=PromptTemplate(
    template='Write a summary for the following candidate -\n {candidate}',
    input_variables=['candidate'])


parser=StrOutputParser()



loader=DirectoryLoader(path='.',glob='*.pdf',loader_cls=PyPDFLoader)
docs1=loader.load()

print(docs1)


chain = prompt | model |parser

chain.invoke({'candidate':docs1[0].page_content})