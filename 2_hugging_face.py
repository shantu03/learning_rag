from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from malicious.ipl_data_cleaning import df,pd
from langchain_core.messages import HumanMessage
csv_as_text='''
Data columns (total 20 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   ID               950 non-null    int64  
 1   City             899 non-null    object 
 2   Date             950 non-null    object 
 3   Season           950 non-null    object 
 4   MatchNumber      950 non-null    object 
 5   Team1            950 non-null    object 
 6   Team2            950 non-null    object 
 7   Venue            950 non-null    object 
 8   TossWinner       950 non-null    object 
 9   TossDecision     950 non-null    object 
 10  SuperOver        946 non-null    object 
 11  WinningTeam      946 non-null    object 
 12  WonBy            950 non-null    object 
 13  Margin           932 non-null    float64
 14  method           19 non-null     object 
 15  Player_of_Match  946 non-null    object 
 16  Team1Players     950 non-null    object 
 17  Team2Players     950 non-null    object 
 18  Umpire1          950 non-null    object 
 19  Umpire2          950 non-null    object 
dtypes: float64(1), int64(1), object(18)
memory usage: 148.6+ KB'''

prompt=f"""
 
YOU HAVE TO GIVE ME PYTHON PANDAS CODE TO GET THE FOLLOWING THINGS 

### IPL DATASET INFO###
{csv_as_text}

### QUESTIONS ###
Task 1: GIVE CODE TO Calculate exactly how many matches each team has played (combining Team1 and Team2). List them.
Task 2: GIVE CODE TO count every time a player appears in a match, and identify the top 3 players with the most played matches.

GIVE PYTHON PANDAS CODE ; AVOID USING COMMENTS ; AVOID USING REASONING IN WORDS DIRECT GIVE CODE ; AVOID USING ANY TEXT OTHER THAN PYTHON CODE
"""
load_dotenv()

llm=HuggingFaceEndpoint(task='text-generation',
                        provider='novita',
                        repo_id='inclusionAI/Ling-3.0-flash-Fin',
                       temperature=0,
                        max_new_tokens=2048)

model = ChatHuggingFace(llm=llm)

generated_code=model.invoke([HumanMessage(content=prompt)]).content
generated_code=generated_code.replace("```python","").replace('```',"").strip()

print(generated_code)
