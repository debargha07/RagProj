import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from ingest import load_and_chunk_pdf
load_dotenv()
def createvector():
    chunks=load_and_chunk_pdf("CSE_Paper_1.pdf")
    embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db="./chromadb.db"
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=db
    )
    return vectorstore
def test_vector_search(vectorstore):
    print("\nStep 4: Testing the Vector Search...")
    # Ask a question related to your sample.pdf
    query = input("Ask the query:")
    
    # Perform a similarity search to find the top 3 most relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"\nTop Match for query: '{query}'")
    print("-" * 40)
    if results:
         print(results[0].page_content)
         print(f"Source: Page {results[0].metadata.get('page', 'Unknown')}")
    else:
         print("No results found.")

if __name__ == "__main__":
    # Create the database and get the connection object
    db = createvector()
    
    # Test that it works
    test_vector_search(db)
