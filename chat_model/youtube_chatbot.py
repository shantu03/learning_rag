from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models.Ling_llm import llm
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from models import Embeddings_Models
from langchain_core.output_parsers import StrOutputParser


loader=YoutubeLoader.from_youtube_url(youtube_url='https://www.youtube.com/watch?v=rDteJAuKiBI')

video_transit=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

docs=text_splitter.split_documents(video_transit)
print(len(docs))


embedding_model=Embeddings_Models.get_mpnet()


vector_store=Chroma.from_documents(documents=docs,
                                   embedding=embedding_model)

retriver=vector_store.as_retriever(search_kwargs={"k":4},search_type='similarity')


prompt=PromptTemplate(template='''YOU ARE GOING TO ANSWER THE QUERY FROM ONLY THIS CONTENT; 
IF YOU CAN\'T FIND ANSWER YOU SIMPLY REPLY THAT YOU DO NOT KNOW ANSWER
query={question}, content={context}
 ''',
input_variables=['context','question'])

parser=StrOutputParser()

from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel


def merge_docs(retrived_docs):
    context_text= '\n\n'.join(doc.page_content for doc in retrived_docs)
    return context_text




parallel_chain=RunnableParallel({
    'context': retriver | RunnableLambda(merge_docs),
    'question':RunnablePassthrough()
})



main_chain=parallel_chain | prompt | llm | parser


while(user:=input('You : '))!='exit':
    print(main_chain.invoke(user))


