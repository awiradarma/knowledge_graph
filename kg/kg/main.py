# For OpenAI

import os

os.environ["OPENAI_API_KEY"] = "INSERT API KEY"
os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"  # replace with your password, by default it is "nebula"
os.environ["NEBULA_ADDRESS"] = "127.0.0.1:9669"  # assumed we have NebulaGraph 3.5.0 or newer installed locally


import logging
import sys
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, ServiceContext, KnowledgeGraphIndex
from llama_index.graph_stores.nebula import NebulaGraphStore

from llama_index.core import download_loader

from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.core import load_index_from_storage
# from llama_hub.youtube_transcript import YoutubeTranscriptReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

Settings.llm = llm
Settings.chunk_size = 512

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


# define LLM
# llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
# service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
#        service_context=service_context,


try:

    # storage_context = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
    kg_index = load_index_from_storage(
        storage_context=storage_context,
        graph_store=graph_store,
        max_triplets_per_chunk=15,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        verbose=True,
    )
    index_loaded = True
except:
    index_loaded = False

if not index_loaded:

    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    wiki_documents = loader.load_data(pages=['Philadelphia Phillies'], auto_suggest=False)
    print(f'Loaded {len(wiki_documents)} documents')

    # youtube_loader = YoutubeTranscriptReader()
    # youtube_documents = youtube_loader.load_data(ytlinks=['https://www.youtube.com/watch?v=k-HTQ8T7oVw'])    
    # print(f'Loaded {len(youtube_documents)} YouTube documents')
    # documents=wiki_documents + youtube_documents,
    print(f'About to build KnowledgeGraphIndex')

    kg_index = KnowledgeGraphIndex.from_documents(
        documents=wiki_documents,
        storage_context=storage_context,
        graph_store=graph_store,
        max_triplets_per_chunk=15,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        # include_embeddings=True,
    )

    print(f'About to persist KnowledgeGraphIndex')
    # kg_index.storage_context.persist(persist_dir='./storage_graph')
    kg_index.storage_context.persist()
    print(f'Finished storing KnowledgeGraphIndex')

query_engine = kg_index.as_query_engine()
response = query_engine.query("Tell me about some of the facts of Philadelphia Phillies.")
display(Markdown(f"<b>{response}</b>"))
