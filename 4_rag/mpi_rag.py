import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from mpi4py import MPI
import numpy as np
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import time

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Enhanced MPI-aware embedding model
class MPIEmbeddings:
    def __init__(self, base_embeddings, comm):
        self.base_embeddings = base_embeddings
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using parallel processing."""
        # Broadcast the full list of texts to all processes
        texts = self.comm.bcast(texts, root=0)
        
        # Split the work among processes
        n_texts = len(texts)
        texts_per_process = n_texts // self.size + (1 if n_texts % self.size > self.rank else 0)
        start_idx = self.rank * (n_texts // self.size) + min(self.rank, n_texts % self.size)
        end_idx = start_idx + texts_per_process
        
        # Each process handles its portion of texts
        local_texts = texts[start_idx:end_idx]
        local_embeddings = []
        
        if local_texts:
            local_embeddings = self.base_embeddings.embed_documents(local_texts)
        
        # Gather results from all processes
        all_embeddings = self.comm.gather((start_idx, local_embeddings), root=0)
        
        if self.rank == 0:
            # Combine and reorder embeddings
            combined_embeddings = [None] * n_texts
            for start_idx, embeddings in all_embeddings:
                for i, embedding in enumerate(embeddings):
                    combined_embeddings[start_idx + i] = embedding
            return combined_embeddings
        else:
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        # This doesn't need parallelization since it's just one query
        return self.base_embeddings.embed_query(text)

# Create base embeddings
base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create MPI-aware embeddings
embeddings = MPIEmbeddings(base_embeddings, comm)

# MPI-aware document processing for ingestion
class MPIDocumentProcessor:
    def __init__(self, comm, embeddings):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.embeddings = embeddings
    
    def batch_process_documents(self, documents: List[Document], batch_size: int = 100) -> List[Dict[str, Any]]:
        """Process documents in batches with MPI parallelization."""
        if self.rank == 0:
            # Only rank 0 has the actual documents
            total_docs = len(documents)
            print(f"Processing {total_docs} documents with {self.size} MPI processes")
        else:
            total_docs = None
        
        # Broadcast total docs to all processes
        total_docs = self.comm.bcast(total_docs, root=0)
        
        # Process documents in batches to avoid memory issues
        all_embeddings = []
        all_metadatas = []
        all_texts = []
        
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            
            if self.rank == 0:
                # Prepare this batch
                batch_docs = documents[batch_start:batch_end]
                batch_texts = [doc.page_content for doc in batch_docs]
                batch_metadatas = [doc.metadata for doc in batch_docs]
                
                print(f"Processing batch {batch_start//batch_size + 1}/{(total_docs+batch_size-1)//batch_size}")
                batch_start_time = time.time()
            else:
                batch_texts = None
                batch_metadatas = None
            
            # Broadcast texts and metadata to all processes
            batch_texts = self.comm.bcast(batch_texts, root=0)
            batch_metadatas = self.comm.bcast(batch_metadatas, root=0)
            
            # Generate embeddings in parallel
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            
            # Only rank 0 collects the results
            if self.rank == 0:
                all_embeddings.extend(batch_embeddings)
                all_metadatas.extend(batch_metadatas)
                all_texts.extend(batch_texts)
                
                batch_end_time = time.time()
                print(f"Batch processed in {batch_end_time - batch_start_time:.2f} seconds")
        
        if self.rank == 0:
            return {"embeddings": all_embeddings, "metadatas": all_metadatas, "texts": all_texts}
        else:
            return None

# Function to add documents to vector store with MPI parallelization
def add_documents_mpi(documents: List[Document], vectorstore):
    """Add documents to the vector store using MPI parallelization for embedding generation."""
    processor = MPIDocumentProcessor(comm, embeddings)
    result = processor.batch_process_documents(documents)
    
    if rank == 0 and result:
        # Only rank 0 adds to the vector store
        vectorstore.add_embeddings(
            texts=result["texts"],
            embeddings=result["embeddings"],
            metadatas=result["metadatas"]
        )
        print(f"Added {len(result['texts'])} documents to vector store")

# Load or create the vector store (only needed by rank 0 for the retrieval part)
if os.path.exists(persistent_directory):
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    # Create a new vector store if it doesn't exist
    if rank == 0:
        print("Creating new vector store - no existing one found")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    # Example code for adding documents (commented out as it's not in the original)
    # from langchain_community.document_loaders import DirectoryLoader
    # documents = DirectoryLoader("your_docs_directory").load()
    # add_documents_mpi(documents, db)

# Create a parallel-enhanced retriever
class MPIEnhancedRetriever:
    def __init__(self, base_retriever, comm):
        self.base_retriever = base_retriever
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
    
    def get_relevant_documents(self, query):
        # If query is a dict (from history-aware retriever), extract the actual query
        if isinstance(query, dict):
            actual_query = query.get("input", "")
        else:
            actual_query = query
            
        # Broadcast the query to all processes
        actual_query = self.comm.bcast(actual_query, root=0)
        
        # Each process will retrieve documents
        if isinstance(query, dict):
            # For history-aware retriever
            if self.rank == 0:
                local_docs = self.base_retriever.get_relevant_documents(query)
            else:
                local_docs = []
        else:
            # For direct query
            local_docs = self.base_retriever.get_relevant_documents(actual_query)
        
        # Gather all documents to rank 0
        all_docs = self.comm.gather(local_docs, root=0)
        
        if self.rank == 0:
            # Flatten the list of documents
            flattened_docs = [doc for sublist in all_docs if sublist for doc in sublist]
            
            # Sort and deduplicate results
            unique_docs = {}
            for doc in flattened_docs:
                doc_id = doc.metadata.get('id', doc.page_content[:50])
                if doc_id not in unique_docs or unique_docs[doc_id].metadata.get('score', 0) < doc.metadata.get('score', 0):
                    unique_docs[doc_id] = doc
            
            # Return top k documents
            sorted_docs = sorted(unique_docs.values(), 
                                key=lambda x: x.metadata.get('score', 0), 
                                reverse=True)
            return sorted_docs[:3]
        else:
            return []

# Only rank 0 needs to create the actual retriever
if rank == 0:
    standard_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3 * size},  # Retrieve more documents since we'll be filtering
    )
else:
    standard_retriever = None

# Create the MPI-enhanced retriever
retriever = MPIEnhancedRetriever(standard_retriever, comm)

# Create a ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Parallelize document processing in the retrieval chain
class MPIRetrievalChain:
    def __init__(self, retriever, qa_chain, comm):
        self.retriever = retriever
        self.qa_chain = qa_chain
        self.comm = comm
        self.rank = comm.Get_rank()
    
    def invoke(self, inputs):
        # Broadcast inputs from rank 0
        if self.rank == 0:
            query = inputs["input"]
            chat_history = inputs["chat_history"]
        else:
            query = None
            chat_history = None
        
        query = self.comm.bcast(query, root=0)
        chat_history = self.comm.bcast(chat_history, root=0)
        
        # Get relevant documents (handled by MPIEnhancedRetriever)
        docs = self.retriever.get_relevant_documents({"input": query, "chat_history": chat_history})
        
        # Only rank 0 processes the documents to generate an answer
        if self.rank == 0:
            # Process documents and answer question
            answer = self.qa_chain.invoke({
                "input": query,
                "chat_history": chat_history,
                "context": docs
            })
            
            return {"answer": answer, "context": docs}
        else:
            return None

# Create MPI-aware retrieval chain
rag_chain = MPIRetrievalChain(history_aware_retriever, question_answer_chain, comm)

# Function to simulate a continual chat (only run by rank 0)
def continual_chat():
    if rank != 0:
        # Non-root processes just wait for commands
        while True:
            cmd = comm.bcast(None, root=0)
            if cmd == "exit":
                break
        return
        
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            # Tell all processes to exit
            comm.bcast("exit", root=0)
            break
            
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        
        # Display the AI's response
        print(f"AI: {result['answer']}")
        
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))
        
        # Broadcast a continue command
        comm.bcast("continue", root=0)

# Function to add new documents with MPI-accelerated embedding generation
def add_new_documents(documents):
    """Add new documents to the vector store with MPI acceleration."""
    if rank == 0:
        print(f"Adding {len(documents)} new documents to the vector store...")
    
    # Use MPI to parallelize embedding generation and add to store
    add_documents_mpi(documents, db)
    
    if rank == 0:
        print("Documents added successfully!")
        # Persist the vector store
        db.persist()

# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()