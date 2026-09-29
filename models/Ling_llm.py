'''
 ling_3.0_flash_model
'''

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(task='text-generation',
                        provider='novita',
                        repo_id='inclusionAI/Ling-3.0-flash-Fin',
                       temperature=0.2,
                        max_new_tokens=2048)

llm = ChatHuggingFace(llm=llm)
