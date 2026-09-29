from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate


chat_template=ChatPromptTemplate([
    ('system','You are xxx'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
# append history

chat_template.invoke({'chat_history':chat_history,'query':"me  "})

