from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformer_local_model import model
em_model=HuggingFaceEmbeddings()

print(em_model.embed_query("hello")- model.encode('hello'))