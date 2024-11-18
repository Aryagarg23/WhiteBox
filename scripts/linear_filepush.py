import os
import re
import ast
import time
import logging
import threading
from PyPDF2 import PdfReader, PdfWriter

from typing import Tuple, List, Optional

import transformers
import torch
import accelerate

#For OLLAMA clas
from langchain.llms.base import LLM
from pydantic import BaseModel
from typing import Optional, List, Any, Mapping

import requests 

from yfiles_jupyter_graphs import GraphWidget
from neo4j import GraphDatabase

from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from langchain_community.graphs.graph_document import Document
from langchain.text_splitter import TokenTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField


from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel
from typing import Optional, List, Any, Mapping
from langchain_core.documents.base import Document
import json


class OllamaLLM(LLM):
    model_name: str
    api_url: str

    def __init__(self, model_name: str, api_url: str = "address"):
        print("Yuh, Bazinga: ", model_name)
        print("Huh?: ", api_url)
        object.__setattr__(self, 'model_name', model_name)
        object.__setattr__(self, 'api_url', api_url)
        print(f"Initialized OllamaLLM with model_name: {self.model_name}")
        print(f"Initialized OllamaLLM with api_url: {self.api_url}")
        self._call(prompt="")

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(f"{self.api_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        return result['response']

    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

class ProcessedDocument:
    def __init__(self, document: Document, title: str = "", summary: str = ""):
        self.metadata = document.metadata.copy()
        self.metadata['title'] = title
        self.metadata['summary'] = summary
        self.page_content = document.page_content

    def to_dict(self):
        return {
            'metadata': self.metadata,
            'page_content': self.page_content
        }

    def to_langchain_document(self) -> Document:
        return Document(metadata=self.metadata, page_content=self.page_content)

    def __repr__(self):
        return f"ProcessedDocument(title={self.metadata['title']!r}, summary={self.metadata['summary']!r}, source={self.metadata.get('source', 'N/A')!r}, page_content_length={len(self.page_content)!r})"

def generate_title(doc: Document, title_model) -> ProcessedDocument:
    # Generate title using the provided model
    title = title_model._call(doc.page_content)
    
    # Create a ProcessedDocument object with the title
    processed_doc = ProcessedDocument(document=doc, title=title)

    return processed_doc

def generate_summary(processed_doc: ProcessedDocument, summary_model) -> ProcessedDocument:
    # Generate summary using the provided model
    summary = summary_model._call(processed_doc.page_content)
    
    # Update the summary in the ProcessedDocument object
    processed_doc.metadata['summary'] = summary

    return processed_doc

def process_documents(documents: List[Document]) -> List[Document]:
    # Instantiate the title model
    title_model = OllamaLLM(model_name="title_maker")

    # First pass: generate titles
    processed_documents = []
    for doc in documents:
        processed_documents.append(generate_title(doc, title_model))
    
    # Instantiate the summary model
    summary_model = OllamaLLM(model_name="summary_maker")

    # Second pass: generate summaries
    final_documents = []
    for proc_doc in processed_documents:
        summary_doc = generate_summary(proc_doc, summary_model)
        final_documents.append(summary_doc.to_langchain_document())
    
    return final_documents

def save_processed_documents(processed_documents: List[Document], file_name: str):
    with open(file_name, 'w') as f:
        json.dump([{'metadata': doc.metadata, 'page_content': doc.page_content} for doc in processed_documents], f, indent=4)

# Setup logging
logging.basicConfig(filename='document_processing_2.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the JSON file and output paths
json_file_path = "../files/graph_documents.json"
output_json_file_path = "../files/all_data_fixed.json"
final_graph_json_file_path = "../files/final_graph_documents.json"

# Set environment variables for Neo4j
NEO4J_URI = 'URI'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = 'PASSWORD'
AURA_INSTANCEID = 'ID'
AURA_INSTANCENAME = 'ID'

os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
graph = Neo4jGraph()

# Load the pre-generated documents from the JSON file
with open(json_file_path, 'r') as json_file:
    filtered_documents = json.load(json_file)

def make_hashable(obj):
    """Recursively convert a dictionary or list into a hashable structure."""
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    return obj

def has_valid_content(doc):
    """Check if the document has a valid summary, title, or text field."""
    try:
        # Use regex to remove 'Node <x> with properties'
        content_str = re.sub(r"Node \d+ with properties", "", doc['page_content']).strip()
        content_dict = ast.literal_eval(content_str)
        valid = any(key in content_dict for key in ['summary', 'title', 'text'])
        return valid
    except (SyntaxError, ValueError) as e:
        logging.error(f"SyntaxError/ValueError in has_valid_content: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Exception in has_valid_content: {str(e)}")
        return False

def serialize_graph_document(graph_document):
    """Convert a GraphDocument object to a JSON-serializable dictionary."""
    return {
        "nodes": [{"type": node.type, "properties": node.properties} for node in graph_document.nodes],
        "relationships": [{"type": rel.type, "properties": rel.properties} for rel in graph_document.relationships]
    }

# Remove duplicates and filter for valid content
seen = set()
unique_documents = []
for doc in filtered_documents:
    if has_valid_content(doc):
        doc_tuple = make_hashable(doc)
        if doc_tuple not in seen:
            seen.add(doc_tuple)
            unique_documents.append(doc)

# Save the processed and deduplicated documents to a new JSON file (optional)
with open(output_json_file_path, 'w') as output_file:
    json.dump(unique_documents, output_file, indent=4)

# Convert JSON documents to langchain documents and then to graph documents
graph_documents = []
llm_transformer = LLMGraphTransformer(llm=Ollama(model="llama3_better"))


# Define a function to sanitize, print invalid nodes, and push valid graph documents to Neo4j
def push_to_neo4j(graph, graph_documents, num):
    sanitized_documents = []
    
    for idx, document in enumerate(graph_documents, num):
        try:
            # Initialize a list for valid nodes within this document
            valid_nodes = []
            
            # Sanitize nodes by skipping those with empty or invalid labels
            for node in document.nodes:
                if node.type and node.type.strip():
                    node.type = node.type.replace('\x00', '')  # Remove null bytes
                    valid_nodes.append(node)
                else:
                    # Log the invalid node
                    logging.warning(f"Skipping invalid node in document {num}: {node}")
            
            # Even if some nodes were invalid, include the document with valid nodes
            document.nodes = valid_nodes
            sanitized_documents.append(document)
            
            # Push sanitized document to Neo4j
            graph.add_graph_documents(
                [document],
                baseEntityLabel=True,
                include_source=True
            )
            
            # Log successful upload
            logging.info(f"Successfully uploaded document {num} to Neo4j.")

        except Exception as e:
            # Log the failure
            logging.error(f"Failed to upload document {num} to Neo4j: {e}")
    
    if not sanitized_documents:
        logging.info("No valid documents to push to Neo4j.")

# Process all documents with indexes greater than 200
threads = []
for idx, doc in enumerate(unique_documents):
    if idx > 200:  # Check if the index is greater than 200
        try:
            content_dict = ast.literal_eval(re.sub(r"Node \d+ with properties", "", doc['page_content']).strip())
            page_content = content_dict.pop('text', None)
            metadata = {**doc['metadata'], **content_dict}
            final_document = Document(page_content=page_content, metadata=metadata)
            graph_docs = llm_transformer.convert_to_graph_documents([final_document])

            # Start a new thread
            thread = threading.Thread(target=push_to_neo4j, args=(graph, graph_docs, idx))
            thread.start()
            threads.append(thread)

        except Exception as e:
            logging.error(f"Error processing document {idx}: {e}")

#tiny change
# Wait for all threads to finish
for thread in threads:
    thread.join()

with open(final_graph_json_file_path, 'a') as final_graph_file:
    json.dump(graph_documents, final_graph_file, indent=4)
    final_graph_file.write("\n")  # Add a newline to separate entries

logging.info("Processing complete.")
