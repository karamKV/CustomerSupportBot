from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI
import jinja2
import json
from typing import Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure OpenAI client
client = AsyncOpenAI(api_key="api_key")

# Setup Jinja2 environment
template_loader = jinja2.FileSystemLoader(searchpath="templateBasePath")
template_env = jinja2.Environment(loader=template_loader)

class QueryInput(BaseModel):
    userQuery: str
    context: str
    answer: str

async def get_openai_response(prompt: str) -> Dict:
    try:
        response = await client.chat.completions.create(
            model="model",  # Changed model name to a valid one
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please provide responses in JSON format with a 'score' field."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        logger.info(f"OpenAI Response: {content}")
        
        parsed_response = json.loads(content)
        if 'score' not in parsed_response:
            logger.warning("Score not found in response")
            parsed_response['score'] = 5  # Default score if not present
            
        return parsed_response
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse OpenAI response")
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def check_relevance(query: str, context: str, answer: str) -> Dict:
    try:
        prompt_template = template_env.get_template("relevance.jinja")
        prompt = prompt_template.render(
            question=query,
            context=context,
        )
        logger.info(f"Relevance prompt: {prompt}")
        result = await get_openai_response(prompt)
        logger.info(f"Relevance result: {result}")
        return result
    except Exception as e:
        logger.error(f"Relevance check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Relevance check error: {str(e)}")

async def check_groundedness(context: str, answer: str) -> Dict:
    try:
        prompt_template = template_env.get_template("groundedness.jinja")
        prompt = prompt_template.render(
            context=context,
            answer=answer
        )
        logger.info(f"Groundedness prompt: {prompt}")
        result = await get_openai_response(prompt)
        logger.info(f"Groundedness result: {result}")
        return result
    except Exception as e:
        logger.error(f"Groundedness check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Groundedness check error: {str(e)}")

@app.post("/evaluate")
async def evaluate_response(input_data: QueryInput):
    try:
        # Run relevance and groundedness checks in parallel
        relevance_task = asyncio.create_task(
            check_relevance(input_data.userQuery, input_data.context, input_data.answer)
        )
        groundedness_task = asyncio.create_task(
            check_groundedness(input_data.context, input_data.answer)
        )

        # Wait for both tasks to complete
        relevance_result, groundedness_result = await asyncio.gather(
            relevance_task, groundedness_task
        )

        response = {
            "relevance": relevance_result.get("score", 5),
            "groundedness": groundedness_result.get("score", 5),
        }
        
        logger.info(f"Final response: {response}")
        return response

    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)