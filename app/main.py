from fastapi import FastAPI, Body, Header, HTTPException, Depends
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

def get_api_key(x_api_key: str = Header(...)):
    if not x_api_key:
        raise HTTPException(status_code=400, detail="API key is missing")
    return x_api_key

@app.post("/test/", response_model=None)
def test(metadata: Dict = Body(...), key: str = Depends(get_api_key)) -> Any:

    llm = ChatOpenAI(api_key=key, temperature=0.0, model="gpt-3.5-turbo-0125")

    prompt_template = """
    Your are a data scientist and data engineer who is an expert in pre-processing data for ML modeling. Be brief and technical in your answers to whatever the input is.\
    The input is: {input}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    output_parser = StrOutputParser()
    
    prompt_chain = (
        {"input": RunnablePassthrough()} 
        | prompt
        | llm
        | output_parser
    )

    input_data = {
        "input": metadata['message'],
    }
    
    try:
        with get_openai_callback() as cb:
            response = prompt_chain.invoke(input_data)
            cost = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost
            }
    except Exception as e:
        response = str(e)
        cb = null
    
    return {"response": response, "cost": cost}


@app.post("/teach/", response_model=None)
def main(metadata: Dict = Body(...), key: str = Depends(get_api_key)) -> Any:
    
    llm = ChatOpenAI(api_key=key, temperature=0.0, model="ft:gpt-3.5-turbo-0125:personal::9G9coX80")
    
    prompt_template = """
    Your task is to search through the json provided "Stats" (descriptive stats of each feature in a dataset) and as a data engineering and data science expert, give precise and specific recommendations one by one for each feature on best practices for transforming or scaling the given feature so that they can be used for ML modeling. The json contains a target_name node which will be the variable we are predicting. You should only return specific best practices based on the stats you see for each feature. 
    
    The feature and stats json are as follows: {stats}
    
    Return specifically in format like\
    feature_name: \
    - your_recommendation to fix skew \
    - your_recommendation to fix scale \
    - your_recommendation to fix outliers \
    - your_recommendation to fix null values \
    
    If nothing should be done to a feature, do not include the feature in your response at all. If nothing for the specific issue should be done, do not include that issue in the bullet list at all. Consider the features in relation to one another, and include a breif general recommendation on normalization, null values, collinearity, negative values (spefifically their impact on loss functions), if anything should be done to the target variable, and given the statistcs - the types of models that would likely be a good fit."
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    output_parser = StrOutputParser()
    
    prompt_chain = (
        {"stats": RunnablePassthrough()} 
        | prompt
        | llm
        | output_parser
    )

    input_data = {
        "stats": metadata,
    }
    
    try:
        with get_openai_callback() as cb:
            response = prompt_chain.invoke(input_data)
            cost = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost
            }
    except Exception as e:
        response = str(e)
        cb = null
    
    return {"response": response, "cost": cost}

