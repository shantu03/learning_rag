from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser

from datetime import datetime
import os 
from langchain_core.prompts import PromptTemplate
msg_hist=[SystemMessage('You are assistant try to keep output minimal ; i set up your max_output_token =2048 so keep answer minimum')]


load_dotenv()
parser=StrOutputParser()

model=ChatHuggingFace(llm=HuggingFaceEndpoint(provider='novita',task='text-generation',repo_id='inclusionAI/Ling-3.0-flash-Fin',max_new_tokens=2048,temperature=0.4))
try:

    while(user_input:=input("You : \n\n"))!='exit':

        msg_hist.append(HumanMessage(user_input))
        result=model.invoke(msg_hist).content
        msg_hist.append(AIMessage(result))
        print(result)
except Exception as e:
    print(e)
finally:
    timestamp=datetime.now().strftime('%m-%d_%H-%M')
    filename=f'chat_history/chat-{timestamp}.txt'

    prompt=PromptTemplate(template='Give me only plain one name for this {text} i want to save it folder -- summaries and think and only one name',
                        input_variables=['text'])

    chain= prompt |model | parser
    with open(filename,'w+',encoding='utf-8') as f:
        for i in msg_hist:
            if isinstance(i,SystemMessage):
                label='System'
            elif isinstance(i,HumanMessage):
                label='You'
            else :
                label='AI'

            f.write(f"[{label}] : {i.content} \n")

    new_name=chain.invoke({'text':msg_hist})
    os.rename(filename,f'chat_history/{new_name}_{timestamp}.txt')



 