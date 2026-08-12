from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import AsyncPinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "facebook"  # Choose your Pinecone index name
async def ingest_docs():
    # Open and read the markdown file content
    with open("politiques_boutique.md", "r", encoding="utf-8") as file:
        md_content = file.read()

    headers_to_split = [("#", "document_title"), ("##", "section_title")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split, strip_headers=False)
    header_splits = md_splitter.split_text(md_content)

    # 3. Sub-chunk long sections (~500 chars)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=12)
    final_chunks = text_splitter.split_documents(header_splits)
    print(f"Total chunks created: {len(final_chunks)}\n")

    print("--- Sample Chunk 1 ---")
    print("CONTENT:\n", final_chunks[0].page_content)
    print("METADATA:\n", final_chunks[0].metadata)
    print("----------------------\n")

    async with AsyncPinecone(api_key=PINECONE_API_KEY) as pc:
        index = await pc.index(INDEX_NAME)
        chunk_texts = [c.page_content for c in final_chunks]
        embeddings_response = await pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=chunk_texts,
            parameters={
                "input_type": "passage",
                "dimension": 384
            }
        )

        # Format vectors for Pinecone
        vectors = []
        for i, (item, chunk) in enumerate(zip(embeddings_response.data, final_chunks)):
            vectors.append((
                f"faq-chunk-{i}",
                item.values,  # 384 floats
                {
                    "text": chunk.page_content,
                    "section": chunk.metadata.get("section_title", "General"),
                    "source": "facebook"
                }
            ))
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        await index.upsert(vectors=vectors, namespace="faq")
        print("Success! Your vectors are now stored in Pinecone.")


if __name__ == "__main__":
    asyncio.run(ingest_docs())