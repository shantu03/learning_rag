from langchain_google_genai import  ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model =ChatGoogleGenerativeAI(model='gemini-3.8-flash')
