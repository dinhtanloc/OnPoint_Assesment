import asyncio
import os
import sys
import time
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from loguru import logger
from utils.rag import cleanup_qdrant_clients

from utils.rag import RAG

st.set_page_config(
    page_title="OnPoint- Document Search Engine",
    page_icon="🤖",
    layout="wide",
)

load_dotenv()


@st.cache_resource
def initialize_rag(
    embedding_type,
    embedding_model,
    enable_hybrid_search,
    chunk_type,
    collection_name,
    persist_dir,
):
    """Cache RAG instance để tránh khởi tạo lại mỗi lần refresh"""
    try:
        rag = RAG(
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_search,
            chunk_type=chunk_type,
            use_memory=False,
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        logger.info(f"RAG initialized with cache")
        return rag
    except Exception as e:
        logger.error(f"RAG initialization error: {e}")
        st.error(f"Error initializing RAG: {e}")
        st.stop()


# ============================== USEFUL FUNCTIONS ==============================


def clear_history():
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.success("History cleared!")


def clear_rag_cache():
    """Clear RAG cache và reinitialize"""
    initialize_rag.clear()
    if "rag" in st.session_state:
        del st.session_state.rag
    if "rag_config_key" in st.session_state:
        del st.session_state.rag_config_key
    cleanup_qdrant_clients()
    st.success("RAG cache cleared! Page will refresh to reinitialize.")
    st.rerun()




def main():
    st.title("🤖 OnPoint- Document Search Engine")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("⚙️ Settings")

        col1, col2 = st.columns(2)
        with col1:
            st.button("Clear history", on_click=clear_history)
        with col2:
            st.button("Clear RAG cache", on_click=clear_rag_cache)

        st.subheader("🗂️ Vector Database Settings")

        embedding_type = st.selectbox(
            "Embedding Type",
            options=["huggingface", "vertexai"],
            index=0,
            help="Choose between Google VertexAI or local HuggingFace embeddings",
        )
        if embedding_type == "huggingface":
            embedding_model = st.selectbox(
                "HuggingFace Model",
                options=[
                    "BAAI/bge-base-en",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ],
                index=0,
                help="Select HuggingFace embedding model",
            )
        else:
            embedding_model = st.selectbox(
                "VertexAI Model",
                options=["text-embedding-004", "textembedding-gecko@003"],
                index=0,
                help="Select Google VertexAI embedding model",
            )
        enable_hybrid_search = st.checkbox(
            "Enable Hybrid Search",
            value=True,
            help="Combine dense and sparse vectors for better retrieval",
        )

        chunk_type = st.selectbox(
            "Text Chunking Strategy",
            options=["character", "recursive"],
            index=0,
            help="Choose text splitting strategy: character (simple) or recursive (smart)",
        )

        collection_name = st.text_input(
            "Collection Name",
            value="OnPoint_db",
            help="Name for the vector database collection",
        )

        persist_dir = "./qdrant_db"

        rag_config_key = f"{embedding_type}_{embedding_model}_{enable_hybrid_search}_{chunk_type}_{collection_name}"

        if (
            "rag" not in st.session_state
            or st.session_state.rag is None
            or "rag_config_key" not in st.session_state
            or st.session_state.rag_config_key != rag_config_key
        ):
            with st.spinner("Initializing Vector Database..."):
                rag = initialize_rag(
                    embedding_type=embedding_type,
                    embedding_model=embedding_model,
                    enable_hybrid_search=enable_hybrid_search,
                    chunk_type=chunk_type,
                    collection_name=collection_name,
                    persist_dir=persist_dir,
                )

            st.session_state.rag = rag
            st.session_state.rag_config_key = rag_config_key
            logger.info(f"RAG initialized with config key: {rag_config_key}")
        else:
            logger.debug("Using existing RAG from session state")

        with st.expander("📊 Current Vector Database Config", expanded=False):
            st.write(f"**Collection:** {collection_name}")
            st.write(f"**Embedding:** {embedding_type} - {embedding_model}")
            st.write(
                f"**Hybrid Search:** {'Enabled' if enable_hybrid_search else 'Disabled'}"
            )
            st.write(f"**Chunk Type:** {chunk_type}")

            if st.session_state.rag:
                sources = st.session_state.rag.get_unique_sources()
                if sources and sources != ["No sources available"]:
                    st.write(f"**Sources:** {len(sources)} document(s)")
                else:
                    st.write("**Sources:** No documents loaded")

        if st.button("🗑️ Clear Database"):
            if st.session_state.rag:
                with st.spinner("Clearing database..."):
                    try:
                        st.session_state.rag.clear_vectorstore()
                        clear_history()
                        st.success("✅ Database cleared!")
                    except Exception as e:
                        st.error(f"❌ Error clearing database: {e}")
            else:
                st.warning("No database to clear")

        st.markdown("---")
        st.subheader("📄 Upload PDF")

        debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Show detailed logging information during PDF processing",
        )

        temp_dir = st.text_input(
            "Custom Temporary Directory (Optional)",
            placeholder="/tmp/pdf_processing",
            help="Specify a custom directory for temporary files. Leave empty to use system default.",
        )

        doc_files = st.file_uploader(
            "Upload PDF", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True
        )

        if doc_files:
            if st.button("🚀 Process PDFs", type="primary"):
                if not st.session_state.rag:
                    st.error(
                        "❌ Vector database not initialized. Please check settings above."
                    )
                    st.stop()

                temp_dir_path = (
                    temp_dir.strip() if temp_dir and temp_dir.strip() else None
                )
                if temp_dir_path and not os.path.exists(temp_dir_path):
                    try:
                        os.makedirs(temp_dir_path, exist_ok=True)
                        st.info(f"📁 Created temporary directory: {temp_dir_path}")
                    except Exception as e:
                        st.error(
                            f"❌ Failed to create temp directory {temp_dir_path}: {e}"
                        )
                        st.stop()

                with st.spinner(f"Processing {len(doc_files)} PDF files..."):
                    try:
                        import concurrent.futures
                        from concurrent.futures import ThreadPoolExecutor

                        from utils.file_loader import PDFLoader, SpreadsheetLoader

                        def process_file_sync(file, loader, debug_mode=False):
                            start_time = time.time()
                            try:
                                file_name = file.name
                                file_ext = os.path.splitext(file_name)[1].lower().strip()
                                logger.info(f"Processing file: {file_name} ({file_ext})")
                                logger.debug(f"Raw file extension: {os.path.splitext(file_name)[1]}")
                                logger.debug(f"Cleaned file extension: {file_ext}")


                                if file_ext in [".pdf"]:
                                    logger.info(f"Processing PDF: {file_name}")
                                    splits = loader.load(pdf_file=file, original_filename=file_name)
                                elif file_ext in [".csv", "xlsx"]:
                                    logger.info(f"Processing spreadsheet: {file_name}")
                                    # Tạo một instance của SpreadsheetLoader riêng biệt cho mỗi file
                                    spreadsheet_loader = SpreadsheetLoader(debug=debug_mode, temp_dir=temp_dir_path)
                                    splits = spreadsheet_loader.load(file, original_filename=file_name)
                                else:
                                    raise ValueError(f"Unsupported file format: {file_ext}")

                                processing_time = time.time() - start_time
                                logger.info(f"Completed {file_name} in {processing_time:.1f}s - {len(splits)} documents")
                                return {
                                    "file_name": file_name,
                                    "success": True,
                                    "splits": splits,
                                    "count": len(splits),
                                    "processing_time": processing_time,
                                    "file_size_mb": len(file.getvalue()) / (1024 * 1024),
                                }
                            except Exception as e:
                                processing_time = time.time() - start_time
                                logger.error(f"Error processing {file.name} after {processing_time:.1f}s: {str(e)}")
                                return {
                                    "file_name": file.name,
                                    "success": False,
                                    "error": str(e),
                                    "splits": [],
                                    "processing_time": processing_time,
                                    "file_size_mb": len(file.getvalue()) / (1024 * 1024),
                                }

                        loader = PDFLoader(
                            debug=debug_mode,
                            temp_dir=temp_dir_path,
                        )

                        all_splits = []
                        results = []

                        start_time = time.time()

                        with ThreadPoolExecutor(max_workers=3) as executor:
                            future_to_file = {
                                executor.submit(
                                    process_file_sync, file, loader, debug_mode
                                ): file
                                for file in doc_files
                            }

                            for future in concurrent.futures.as_completed(
                                future_to_file
                            ):
                                result = future.result()
                                results.append(result)

                                if result["success"]:
                                    all_splits.extend(result["splits"])

                        total_time = time.time() - start_time
                        successful_files = len([r for r in results if r["success"]])
                        total_docs = sum(r["count"] for r in results if r["success"])
                        text_docs = len(
                            [
                                doc
                                for doc in all_splits
                                if doc.metadata.get("type") == "text"
                            ]
                        )
                        table_docs = len(
                            [
                                doc
                                for doc in all_splits
                                if doc.metadata.get("type") == "table"
                            ]
                        )

                        stats = {
                            "successful_files": successful_files,
                            "total_files": len(doc_files),
                            "total_docs": total_docs,
                            "text_docs": text_docs,
                            "table_docs": table_docs,
                            "total_time": total_time,
                        }

                        if all_splits:
                            # Add documents to vectorstore
                            try:
                                with st.spinner(
                                    "Adding documents to vector database..."
                                ):
                                    st.session_state.rag.add_documents(
                                        documents=all_splits
                                    )

                                st.success(
                                    f"🎉 Successfully processed {stats['successful_files']}/{stats['total_files']} PDF(s)!\n\n"
                                    f"📄 **Total documents:** {stats['total_docs']} "
                                    f"({stats['text_docs']} text, {stats['table_docs']} tables) | "
                                    f"⏱️ **Time:** {st.session_state.rag._format_time(stats['total_time'])}\n\n"
                                    f"✅ **Added to vector database:** {len(all_splits)} documents"
                                )

                            except Exception as e:
                                st.error(
                                    f"❌ Error adding documents to vector database: {e}"
                                )
                                logger.error(f"Vectorstore add_documents error: {e}")

                        else:
                            st.error(
                                "❌ No documents were extracted from the PDF files."
                            )

                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {e}")
                        logger.error(f"PDF processing error: {e}")

    st.markdown("---")

    chat_col, context_col = st.columns([2, 1])  # 2:1 ratio

    with chat_col:
        st.subheader("🗨️ Conversation")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    with context_col:
        st.subheader("📋 Retrieved Context")

        if not st.session_state.messages:
            st.info("💡 Start a conversation to see relevant documents here!")
        else:
            st.info("💡 Context will appear here when asking questions!")

        if st.session_state.rag:
            sources = st.session_state.rag.get_unique_sources()
            if sources and sources != ["No sources available"]:
                st.markdown("### 📚 Available Sources:")
                for i, source in enumerate(sources[:10], 1):
                    st.write(f"{i}. {source}")
                if len(sources) > 10:
                    st.write(f"... and {len(sources) - 10} more documents")
            else:
                st.write("📭 No documents in database yet")

    if prompt := st.chat_input("Ask me anything about your documents!"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Searching knowledge base..."):
                try:
                    if st.session_state.rag:
                        rag_result = st.session_state.rag(prompt)

                        docs = rag_result["docs"]
                        context = rag_result["context"]
                        response = rag_result["response"]
                        query = rag_result["query"]

                        if docs:
                            response = docs[1].page_content

                        with context_col:
                            st.markdown("### 🔍 Retrieved for Current Query:")

                            if docs:
                                with st.expander("📄 Source Documents", expanded=True):
                                    st.write(
                                        f"**Total Retrieved: {len(docs)} documents**"
                                    )
                                    st.markdown("---")

                                    for i, doc in enumerate(docs, 1):
                                        source = doc.metadata.get(
                                            "source", "Unknown source"
                                        )
                                        page = doc.metadata.get("page", "Unknown page")
                                        doc_type = doc.metadata.get("type", "text")

                                        # Show source info with page
                                        st.write(f"**{i}. {source}** (Page {page})")

                                        # Type badge
                                        if doc_type == "table":
                                            st.markdown("🔢 `TABLE`")
                                        else:
                                            st.markdown("📝 `TEXT`")

                                        # Content preview
                                        preview = (
                                            doc.page_content[:300] + "..."
                                            if len(doc.page_content) > 300
                                            else doc.page_content
                                        )
                                        st.markdown(f"*{preview}*")

                                        if i < len(docs):
                                            st.markdown("---")

                                if st.checkbox(
                                    "🔍 Show Full Context",
                                    key=f"show_context_{len(st.session_state.messages)}",
                                ):
                                    with st.expander(
                                        "📋 Full Context Sent to LLM", expanded=False
                                    ):
                                        st.text(
                                            context[:2000] + "..."
                                            if len(context) > 2000
                                            else context
                                        )
                            else:
                                st.info("No relevant documents found for this query.")

                    else:
                        with context_col:
                            st.info("📤 Upload PDF documents to start searching!")
                        response = "Please upload PDF documents first to start using the knowledge base."

                except Exception as e:
                    logger.error(f"RAG processing error: {e}")
                    with context_col:
                        st.error(f"Search error: {str(e)}")
                    response = f"Error processing query: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response})

        with chat_col:
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    main()
