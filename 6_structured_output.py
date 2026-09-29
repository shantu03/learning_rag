from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. Define the Pydantic schema
class Review(BaseModel):
    key_themes: str = Field(description='write all the key themes discussed in review')
    summary: str = Field(description='write down the summary of review in 4 sentences')
    sentiment: Literal['Positive', 'Negative'] = Field(description='return sentiment of review')
    pros: Optional[list[str]] = Field(default=None, description='write down pros')
    cons: Optional[list[str]] = Field(default=None, description='write down cons')
    name: Optional[str] = Field(default="guest", description='write down the name of reviewer')

# 2. Instantiate the parser to format prompts and handle output text
parser = JsonOutputParser(pydantic_object=Review)

# 3. Create a template that includes the required format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that extracts structured data. "
               "You MUST strictly follow these formatting guidelines:\n{format_instructions}"),
    ("human", "Extract the review details from this text:\n\n{review_text}")
])

# 4. Initialize the LLM (Lower temperature improves JSON formatting accuracy)
llm = HuggingFaceEndpoint(
    provider='novita',
    task='text-generation',
    repo_id='inclusionAI/Ling-3.0-flash-Fin',
    max_new_tokens=2048,
    temperature=0.1 
)
model = ChatHuggingFace(llm=llm)

# 5. Build the LangChain Expression Language (LCEL) chain
chain = prompt | model | parser

# Input text to extract
review_content = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh"""

# 6. Run the chain, feeding the format instructions dynamically
try:
    result = chain.invoke({
        "review_text": review_content,
        "format_instructions": parser.get_format_instructions()
    })
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")




# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import Literal

# load_dotenv()

# model = ChatOpenAI()

# parser = StrOutputParser()

# class Feedback(BaseModel):

#     sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# parser2 = PydanticOutputParser(pydantic_object=Feedback)

# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
#     input_variables=['feedback'],
#     partial_variables={'format_instruction':parser2.get_format_instructions()}
# )

# classifier_chain = prompt1 | model | parser2

# prompt2 = PromptTemplate(
#     template='Write an appropriate response to this positive feedback \n {feedback}',
#     input_variables=['feedback']
# )

# prompt3 = PromptTemplate(
#     template='Write an appropriate response to this negative feedback \n {feedback}',
#     input_variables=['feedback']
# )

# branch_chain = RunnableBranch(
#     (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
#     (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
#     RunnableLambda(lambda x: "could not find sentiment")
# )

# chain = classifier_chain | branch_chain

# print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# chain.get_graph().print_ascii()