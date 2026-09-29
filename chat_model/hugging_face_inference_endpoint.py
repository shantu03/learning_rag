from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id='inclusionAI/Ling-3.0-flash-Fin',
                        provider='novita',
                        task='text-generation')

model=ChatHuggingFace(llm=llm)

print(model.invoke('what the latest info this model have ? give date approx i just want an rought idea ').content)