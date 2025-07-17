import os
from contextlib import asynccontextmanager
from typing import List, Optional, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse

from pinecone import Pinecone, ServerlessSpec

from llama_index.core.tools import QueryEngineTool

from llama_index.core.agent import AgentRunner

from ingest_data import load_and_parse_multimodal_documents, validate_multimodal_setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
pinecone_index = None
vector_store = None
agent: Optional[AgentRunner] = None
parser = None
index: Optional[VectorStoreIndex] = None

class QueryRequest(BaseModel):
    query_text: str

class QueryResponse(BaseModel):
    response: str
    source_nodes: Optional[List[dict]] = None

class IndexResponse(BaseModel):
    message: str
    agent_available: bool
    documents_processed: int
    parsing_methods: dict
    multimodal_info: dict

def init():
    global pinecone_index, vector_store, parser, index
    try:
        required_vars = ["GOOGLE_API_KEY", "LLAMA_CLOUD_API_KEY", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Config LlamaIndex
        Settings.llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
        )
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="models/embedding-001",
            api_key=GOOGLE_API_KEY,
        )
        Settings.chunk_size = 512

        # config LlamaParse
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            user_prompt=(
                "Extract all content from this document including text, tables, charts, and graphs. "
                "Pay special attention to numerical data, financial metrics, and economic indicators. "
                "Preserve the structure and context of tables and charts. "
                "If you encounter charts or graphs, describe them in detail including axis labels, data points, and trends."
            ),
            num_workers=4,
            verbose=True,
            check_interval=10,
            max_timeout=60
        )

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "multimodal-rag-demo"
        existing_indexes = pc.list_indexes()

        index_exists = index_name in existing_indexes
        if not index_exists:
            logger.info(f"Creating Pinecone index: {index_name}")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=3072,  # Gemini embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            except Exception as e:
                # If index already exists, ignore the error
                if hasattr(e, 'status') and getattr(e, 'status', None) == 409:
                    logger.warning(f"Pinecone index '{index_name}' already exists. Skipping creation.")
                elif 'ALREADY_EXISTS' in str(e):
                    logger.warning(f"Pinecone index '{index_name}' already exists. Skipping creation.")
                else:
                    logger.error(f"Error creating Pinecone index: {str(e)}")
                    raise
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Try to load index from vector store if possible
        try:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            logger.info("Loaded index from vector store.")
        except Exception as e:
            logger.warning(f"Could not load index from vector store: {str(e)}")
            index = None

        logger.info("All components initialized successfully with multimodal support")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

def init_agent():
    global agent, index
    try:
        tools = []
        
        # Add RAG tool if index is available
        if index is not None:
            rag_query_engine = index.as_query_engine(similarity_top_k=3)
            rag_tool = QueryEngineTool.from_defaults(
                query_engine=rag_query_engine,
                name="document_search",
                description="Search through indexed documents to find relevant information about economic data, capital formation, and other topics from the document collection."
            )
            tools.append(rag_tool)
            logger.info("RAG tool added to agent.")
        
        # Add Tavily tool if API key is available
        if TAVILY_API_KEY:
            try:
                from llama_index.tools.tavily_research.base import TavilyToolSpec
                tavily_tool = TavilyToolSpec(api_key=TAVILY_API_KEY)
                tavily_tools = tavily_tool.to_tool_list()
                tools.extend(tavily_tools)
                logger.info("Tavily search tool added to agent.")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily tool: {e}")
        
        if tools:
            agent = AgentRunner.from_llm(
                llm=Settings.llm, 
                tools=tools, 
                verbose=True
            )
            logger.info(f"Agent initialized with {len(tools)} tools.")
        else:
            agent = None
            logger.warning("Agent not initialized: no tools available.")
            
    except Exception as e:
        logger.error(f"Agent initialization failed: {str(e)}")
        agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    init()
    init_agent()
    yield

app = FastAPI(
    title="Multimodal RAG with Image Analysis",
    description="RAG system with text and image understanding",
    version="3.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Ready"}

@app.post("/index", response_model=IndexResponse)
async def index_documents():
    global index
    try:
        if not all([parser, vector_store]):
            raise HTTPException(status_code=500, detail="Components not initialized")

        data_dir = "data"
        if not os.path.exists(data_dir):
            raise HTTPException(status_code=404, detail=f"Data directory '{data_dir}' not found")

        logger.info("Starting multimodal document indexing...")

        # Load documents with multimodal processing
        documents = load_and_parse_multimodal_documents(
            data_dir,
            parser,
            os.getenv("GOOGLE_API_KEY")
        )

        if not documents:
            raise HTTPException(status_code=404, detail="No documents processed successfully")

        # Analyze processing results
        parsing_methods = {}
        multimodal_info = {
            "text_documents": 0,
            "image_analyses": 0,
            "multimodal_documents": 0,
            "total_images_processed": 0
        }

        for doc in documents:
            method = doc.metadata.get("parsing_method", "unknown")
            parsing_methods[method] = parsing_methods.get(method, 0) + 1
            if "multimodal" in method:
                multimodal_info["multimodal_documents"] += 1
            elif "image" in method:
                multimodal_info["image_analyses"] += 1
                multimodal_info["total_images_processed"] += 1
            else:
                multimodal_info["text_documents"] += 1

        # Create index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        logger.info(f"Successfully indexed {len(documents)} documents with multimodal processing")

        # Re-initialize agent after re-indexing
        init_agent()

        return IndexResponse(
            message="Documents indexed with multimodal image analysis",
            agent_available=agent is not None,
            documents_processed=len(documents),
            parsing_methods=parsing_methods,
            multimodal_info=multimodal_info
        )

    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    global agent, index
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")

        logger.info(f"Processing multimodal query: {request.query_text}")

        # Ensure agent is initialized
        if agent is None:
            logger.info("Agent not initialized, attempting to initialize.")
            init_agent()
            if agent is None:
                raise HTTPException(status_code=400, detail="Agent not initialized. Please check index and try again.")

        # Use aquery for async operation
        response = await agent.aquery(request.query_text)
        answer = str(response)

        # Process source nodes with multimodal metadata
        source_nodes = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                node_text = getattr(node, 'text', str(node))
                if isinstance(node_text, str):
                    node_text = node_text.strip()
                if len(node_text) > 500:
                    node_text = node_text[:500] + "..."
                # Defensive: node.metadata may be dict or object
                metadata = getattr(node, 'metadata', {})
                if not isinstance(metadata, dict):
                    try:
                        metadata = dict(metadata)
                    except Exception:
                        metadata = {}
                source_info = {
                    "score": float(getattr(node, 'score', 0)) if hasattr(node, 'score') else None,
                    "text": node_text,
                    "metadata": {
                        "file_name": metadata.get("file_name", "unknown"),
                        "file_type": metadata.get("file_type", "unknown"),
                        "parsing_method": metadata.get("parsing_method", "unknown"),
                        "confidence_score": metadata.get("confidence_score"),
                        "image_description": metadata.get("image_description"),
                        "page_number": metadata.get("page_number")
                    }
                }
                source_nodes.append(source_info)

        # Sort by relevance score
        source_nodes.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info(f"Query completed with {len(source_nodes)} source nodes")

        return QueryResponse(
            response=answer,
            source_nodes=source_nodes[:8]  # Return top 8 sources
        )

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/health")
async def health_check():

    multimodal_status = validate_multimodal_setup()
    
    return {
        "status": "healthy" if multimodal_status else "degraded",
        "components": {
            "parser": parser is not None,
            "vector_store": vector_store is not None,
            "pinecone_index": pinecone_index is not None,
            "multimodal_processing": multimodal_status
        },
        "version": "3.0.0",
        "features": ["text_extraction", "image_analysis", "multimodal_processing"]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)