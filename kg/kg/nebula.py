# For OpenAI

import os

os.environ["OPENAI_API_KEY"] = "INSERT API KEY"
os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"  # replace with your password, by default it is "nebula"
os.environ["NEBULA_ADDRESS"] = "127.0.0.1:9669"  # assumed we have NebulaGraph 3.5.0 or newer installed locally


import logging
import sys
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore

from IPython.display import Markdown, display

documents = SimpleDirectoryReader("data/paul_graham").load_data()

space_name = "phillies_rag"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

print(f'About to build KnowledgeGraphIndex')

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

query_engine = index.as_query_engine()
response = query_engine.query("Tell me more about Interleaf")
