from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict, Any
from fastapi.responses import JSONResponse
from langchain.vectorstores import FAISS
import jinja2
import json
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# Initialize OpenAI client
api_key = "api_key"
client = OpenAI(api_key=api_key)

# Initialize FAISS
docsearch = FAISS.load_local(
    "retrieverPath",
    OpenAIEmbeddings(api_key=api_key, model="embeddingModel"),
    allow_dangerous_deserialization=True
)
retriever = docsearch.as_retriever(search_kwargs={"k": 10})

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Jinja template
template_loader = jinja2.FileSystemLoader(searchpath="templateBasePath")
template_env = jinja2.Environment(loader=template_loader)
prompt_template = template_env.get_template("prompt.jinja")

class QueryRequest(BaseModel):
    userQuery: str

def format_chunks(chunks: List[Any]) -> str:
    formatted_text = ""
    for idx, chunk in enumerate(chunks, 1):
        formatted_text += f"\nCHUNK ID {idx}:\n{chunk.page_content}\n"
    return formatted_text

def clean_metadata(metadata):
    """Clean metadata to ensure JSON serializable values"""
    if isinstance(metadata, dict):
        return {k: clean_metadata(v) for k, v in metadata.items()}
    elif isinstance(metadata, list):
        return [clean_metadata(item) for item in metadata]
    elif isinstance(metadata, (int, str, bool)):
        return metadata
    elif isinstance(metadata, float):
        if not float('inf') >= metadata >= float('-inf'):
            return str(metadata)
        return metadata
    else:
        return str(metadata)

def get_metadata_for_references(reference_ids: List[str], chunks: List[Any]):
    """Get metadata for referenced chunks"""
    ref_list = []
    for id_ in reference_ids:
        id_ = int(id_) - 1
        ref = clean_metadata(chunks[id_].metadata)
        ref_list.append(ref)
    return ref_list

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        # Get chunks and format context
        chunks = retriever.invoke(request.userQuery)
        formatted_context = format_chunks(chunks)
        
        # Prepare prompt
        prompt = prompt_template.render(
            userQuery=request.userQuery,
            context=formatted_context
        )
        
        # Use the new OpenAI client
        response = client.chat.completions.create(
            model="model",  # Fixed model name
            messages=[
                {"role": "system", "content": "You are an experienced expert, skilled in answering questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        answer_json = json.loads(response.choices[0].message.content)
        
        # Extract references and get metadata
        references = answer_json.get('reference', [])
        metadata = get_metadata_for_references(references, chunks)
        
        response_data = {
            "status": "success",
            "data": {
                "answer": answer_json.get('answer', ''),
                "relevancy_reason": answer_json.get('relevancy_reason', ''),
                "references": [f"CHUNK ID {ref}" for ref in references],
                "metadata": metadata,
                "context":formatted_context
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "error": {
                    "message": str(e),
                    "type": type(e).__name__
                }
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,timeout_keep_alive=60)
