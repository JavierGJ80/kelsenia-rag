# Imports
import os
import asyncio
import typer
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
import datetime
import json

from dotenv import load_dotenv

from llama_index.llms.anthropic import Anthropic
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.llms.langchain import LangChainLLM
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.base.response.schema import Response
from llama_index.core.utils import truncate_text
from llama_index.core.bridge.pydantic import SerializeAsAny
from llama_index.core.schema import BaseNode, NodeWithScore

from langchain_openai import OpenAI, ChatOpenAI

from milvus_index import create_index
from prompt_templates import TEXT_QA_TEMPLATE, REFINE_TEMPLATE, TEXT_QA_TEMPLATE_DEFAULT
from utils import create_tuple_from_pdf_dir


app = FastAPI()
cli = typer.Typer()

# At the module level
reranker: Optional[FlagEmbeddingReranker] = None
index = None


load_dotenv()


@cli.command()
def run_server(
    milvus_db_path: str = typer.Option(
        "milvus_demo.db", help="Path to the Milvus database"
    ),
    collection_name: str = typer.Option(
        "hybrid_pipeline", help="Name of the collection"
    ),
    reload_docs: bool = typer.Option(True, help="Whether to reload documents"),
    file_paths: list[str] = typer.Option(
        ["explorationDocs/2q24-cfsu-1.pdf"], help="Paths to the files to index"
    ),
):
    global reranker
    reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10)
    global index
    print('Asingnin index')
    index = create_index(
        milvus_db_path=milvus_db_path,
        collection_name=collection_name,
        reload_docs=reload_docs,
        file_paths=tuple(file_paths),
    )
    print('Runing uvicorn app')
    uvicorn.run(app, host="0.0.0.0", port=8001)

@cli.command()
def create_index_db(
        milvus_db_path: str = typer.Option(
        "milvus_demo.db", help="Path to the Milvus database"
    ),
    collection_name: str = typer.Option(
        "hybrid_pipeline", help="Name of the collection"
    ),
    directory: str = typer.Option(
        "explorationDocs/twoTopicsDb", help="Path to the directory to index"
    ),
):
    start = time.perf_counter()
    global reranker
    reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10)
    global index
    print(f"\033[92mInformation:\033[0m \033[95mAssigning index.\033[0m")
    index = create_index(
        milvus_db_path=milvus_db_path,
        collection_name=collection_name,
        reload_docs=True,
        file_paths=create_tuple_from_pdf_dir(directory),
    )
    end = time.perf_counter()
    total_time = end - start
    time_with_format = str(datetime.timedelta(seconds=total_time))
    print(f"\033[92mInformation:\033[0m \033[95mThe indexing process completed in {time_with_format}.\033[0m")

def create_query_engine(index, languageModel):
    if languageModel == "openai":
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            timeout=600,
        )
        llm = LangChainLLM(llm)
        prompts = {
            "text_qa_template": PromptTemplate(TEXT_QA_TEMPLATE_DEFAULT),
        }
    elif languageModel == "claude":
        llm = Anthropic(
            model="claude-3-5-sonnet-20240620",
        )
        prompts = {
            "text_qa_template": PromptTemplate(TEXT_QA_TEMPLATE_DEFAULT),
        }
    elif languageModel == "granite":
        llm = OpenAI(
            openai_api_base=f"http://localhost:8000/v1",
            # openai_api_base=f"https://cf47-52-117-121-50.ngrok-free.app/v1",
            model="/new_data/experiments/ss-bnp-p10/hf_format/samples_2795520",
            temperature=0.0,
            timeout=600,
        )
        llm = LangChainLLM(llm)
        prompts = {
            "text_qa_template": PromptTemplate(TEXT_QA_TEMPLATE),
            "refine_template": PromptTemplate(REFINE_TEMPLATE),
        }
    else:
        raise ValueError(f"model {languageModel} not supported")

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        node_postprocessors=[reranker],
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        response_mode=ResponseMode.REFINE,
        **prompts,
    )

    return query_engine


class Query(BaseModel):
    llm: str
    query_str: str

def parse_node(node_with_score: NodeWithScore):
    parsed_node = {
        'filename': node_with_score.node.metadata.get('filename', 'no_filename'),
        'content': truncate_text(node_with_score.node.get_content().strip(), 350),
        'score': node_with_score.score
    }
    return parsed_node

def parse_query_res(query_res: Response):
    nodes = [parse_node(node_with_score) for node_with_score in query_res.source_nodes]
    json_response = {
        'response': query_res.response,
        'nodes': nodes
    }

    return json_response

async def get_response(query: Query):
    languageModel = query.llm
    query_str = query.query_str
    global index
    query_engine = create_query_engine(index, languageModel)

    # Execute the query
    query_res: Response = await query_engine.aquery(query_str)
    json_response = parse_query_res(query_res)

    print(
        f"\033[92mResponse from {languageModel} for question:\033[0m \033[95m'{query_str}'\033[0m\n\033[0m{query_res}\033[0m"
    )
    return json_response


@app.post("/get_response")
async def get_response_endpoint(query: Query):
    try:
        json_response = await get_response(query)
        return JSONResponse(content=json_response)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


import inspect


def print_calling_line():
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function_name = frame.f_code.co_name
    print(
        f"\033[91mFunction '{function_name}' called from {filename}, line {lineno}\033[0m"
    )


def debug_get_response(query: Query):
    global reranker
    reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10)
    global index

    index = create_index(
        milvus_db_path="/home/lab/milvus_demo.db",
        collection_name="hybrid_pipeline",
        reload_docs=False,
        file_paths=("/new_data/aldo/rag/2q24-cfsu-1.pdf",),
    )
    languageModel = query.llm
    query_str = query.query_str
    query_engine = create_query_engine(index, languageModel)

    # Execute the query
    query_res = query_engine.query(query_str)
    print(
        f"\033[92mResponse from {languageModel} for question:\033[0m \033[95m'{query_str}'\033[0m\n\033[0m{query_res}\033[0m"
    )
    return query_res.response  # Return just the response string


if __name__ == "__main__":
    cli()
    # query = Query(llm="granite", query_str="What is the net income attributable to equity holders for the first half of 2024?")
    # response = debug_get_response(query)
