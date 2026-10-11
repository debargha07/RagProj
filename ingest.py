from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path):
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    
    print(f"Split document into {len(chunks)} chunks.")
    print("\n--- Sample Chunk (Chunk 3) ---")
    if len(chunks) > 2:
        print(chunks[1].page_content)
        print(f"\nMetadata: {chunks[1].metadata}")
    else:
        print("Document too short for a 3rd chunk.")

    return chunks

if __name__ == "__main__":
    my_chunks = load_and_chunk_pdf("CSE_Paper_1.pdf")