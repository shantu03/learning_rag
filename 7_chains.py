from models.Ling_llm import llm

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

'''
# Seqential Chain 

prompt=PromptTemplate(template='Generate detailed summary of topic : {topic} in approx 1500 words ; explain like you are explaining to child',
                        input_variables=['topic'])

parser=StrOutputParser()

prompt2=PromptTemplate(template='generate 5 point summary from following text : {text}',
                       input_variables=["text"])



chain= prompt | model | parser | prompt2 | model | parser

print(chain.invoke({'topic':input('Enter Topic : ')}))

'''

# Parallel Chain -> not done

# Conditional Chain
''''''
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from pydantic import BaseModel,Field
from typing import Literal


class Senti(BaseModel):
    sentiment :Literal['pos','neg'] =Field(description='Give the sentiment of Review ')

parser1= StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Senti)

prompt1=PromptTemplate(template='Give the sentiment this review :{review} in this format{format}',
                       input_variables=['review'],
                       partial_variables={'format':parser2.get_format_instructions()})


chain1= prompt1 | llm | parser2 

prompt2=PromptTemplate(template='write an appropriate response in style of 1980s to this negative feedback {feedback}',
                       input_variables=['feedback'])

prompt3=PromptTemplate(template='write an appropriate in GenZ style response to this positive feedback {feedback}'
                       ,input_variables=['feedback'])



branch_chain=RunnableBranch(
    (lambda x : x.sentiment=='pos',(lambda x : {'feedback':x['review']}) | prompt3 | llm |parser1),
    (lambda x : x.sentiment=='neg',(lambda x : {'feedback':x['review']}) | prompt2 | llm |parser1),
    RunnableLambda(lambda x : 'sentiment could not find ')

)

final_chain = (
    RunnableParallel(
        sentiment=chain1,
        review=lambda x: x['review']
    ) 
    | branch_chain
)

text='this is terrible  product '
result=final_chain.invoke({'review':text})


print(result)

chain1.get_graph().print_ascii()