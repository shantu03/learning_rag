from sentence_transformers import SentenceTransformer

sentences=['this is one ']

model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print(model.encode(sentences))