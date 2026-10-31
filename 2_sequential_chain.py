from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT']="sequantial-llm-app"


load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0.7,model_kwargs={
        "reasoning_effort": "none"
        })
model2 = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0.5,model_kwargs={
        "reasoning_effort": "none"
        })

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config ={
    'run_name':'sequential-chain',
    'tags':['llm','sequential_chain'],
    'metadata':{
        'model':'gpt-5.6-luna',
        'description':'A sequential chain that generates a report and then summarizes it.'
    }
}

result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
