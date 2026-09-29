

from langchain_core.runnables import RunnableLambda,RunnableBranch,RunnablePassthrough,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.prompts import PromptTemplate

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from models.Ling_llm import llm

prompt1=PromptTemplate(template='write a detailed report on {topic}',
                       input_variables=['topic'])

prompt2=PromptTemplate(template='generate the summary of {topic_data}',
                       input_variables=['topic_data'])

parser=StrOutputParser()


report_gen_chain=RunnableSequence(prompt1,llm,parser)

branch_chain=RunnableBranch(
    (lambda x : len(x.split())>500,RunnableSequence(prompt2,llm,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'india vs pakistan'}))


