from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


docs=[  
    Document(page_content='India\'s Got Latent was launched on June 14, 2024, by Samay Raina on his YouTube channel'),
    Document(page_content='In June 2026, speculation about the release of the second season of India\'s Got Latent increased after Netflix India shared a promotional post featuring Raina\'s bodyguard'),
    Document(page_content='The unique aspect of India\'s Got Latent is its contestant rating system. Before each performance, contestants have to rate themselves. If the average rating given by the judge panel after the performance matches the contestant\'s self-rating, the contestant wins'),
    Document(page_content='New episodes were announced to be released every two weeks on both platforms.')
]

embedding_model=HuggingFaceEmbeddings()

vectorStores=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name='my_collection'
)

retriver=vectorStores.as_retriever(search_kwargs={'k':2})

# print(retriver.invoke('weeks'))

retriver=WikipediaRetriever(top_k_results=1,lang='en')

result=retriver.invoke('Dr Vaibhav Soni')
print(result[0].page_content)