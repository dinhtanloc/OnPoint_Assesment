import asyncio
import atexit
import logging
import os
import time
from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from .file_loader import PDFLoader
from .prompts import GENERATE_PROMPT
from .vectorstore import QdrantClientManager, VectorStore


def cleanup_qdrant_clients():
    """Cleanup function to close all Qdrant clients on app shutdown."""
    try:
        QdrantClientManager.close_all_clients()
        logger.info("All Qdrant clients closed successfully")
    except Exception as e:
        logger.warning(f"Error during Qdrant cleanup: {e}")


atexit.register(cleanup_qdrant_clients)


class RAG:
    def __init__(
        self,
        embedding_type: str,
        embedding_model: str,
        enable_hybrid_search: bool,
        chunk_type: str,
        use_memory: bool,
        collection_name: str,
        persist_dir: str,
    ):
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.enable_hybrid_search = enable_hybrid_search
        self.chunk_type = chunk_type
        self.use_memory = use_memory
        self.collection_name = collection_name

        self.vectorstore_key = f"vs_{embedding_type}_{embedding_model}_{enable_hybrid_search}_{chunk_type}_{use_memory}_{collection_name}"
        self.vectorstore = VectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_search,
            chunk_type=chunk_type,
            use_memory=use_memory,
        )

    def _format_time(self, seconds):
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"

    async def _process_pdf_async(self, pdf_file_path, loader, debug_mode=False):
        """Asynchronous wrapper for PDF processing"""
        start_time = time.time()
        try:
            logger.info(f"Starting processing {pdf_file_path}")

            loop = asyncio.get_event_loop()
            splits = await loop.run_in_executor(
                None,
                lambda: loader.load(
                    path_string=pdf_file_path,
                    original_filename=os.path.basename(pdf_file_path),
                ),
            )

            processing_time = time.time() - start_time

            logger.info(
                f"Completed {pdf_file_path} in {processing_time:.1f}s - {len(splits)} documents"
            )

            return {
                "file_name": os.path.basename(pdf_file_path),
                "success": True,
                "splits": splits,
                "count": len(splits),
                "processing_time": processing_time,
                "file_size_mb": os.path.getsize(pdf_file_path) / (1024 * 1024),
            }
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Error processing {pdf_file_path} after {processing_time:.1f}s: {str(e)}"
            )
            return {
                "file_name": os.path.basename(pdf_file_path),
                "success": False,
                "error": str(e),
                "splits": [],
                "processing_time": processing_time,
                "file_size_mb": os.path.getsize(pdf_file_path) / (1024 * 1024),
            }

    async def load_pdfs(
        self,
        pdf_files: List[str],
        temp_dir: str = None,
        debug_mode: bool = False,
    ):
        """Process multiple PDF files asynchronously"""
        loader = PDFLoader(
            debug=debug_mode,
            temp_dir=temp_dir,
            enrich=False,
        )

        all_splits = []

        start_time = time.time()

        tasks = [
            self._process_pdf_async(pdf_file_path, loader, debug_mode)
            for pdf_file_path in pdf_files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "file_name": "unknown",
                        "success": False,
                        "error": str(result),
                        "splits": [],
                        "processing_time": 0,
                        "file_size_mb": 0,
                    }
                )
            else:
                processed_results.append(result)
                if result["success"]:
                    all_splits.extend(result["splits"])

        total_time = time.time() - start_time
        successful_files = len([r for r in processed_results if r["success"]])
        total_docs = sum(r["count"] for r in processed_results if r["success"])
        text_docs = len(
            [doc for doc in all_splits if doc.metadata.get("type") == "text"]
        )
        table_docs = len(
            [doc for doc in all_splits if doc.metadata.get("type") == "table"]
        )

        logger.info(
            f"Processing completed: {successful_files}/{len(pdf_files)} files, {total_docs} documents, {self._format_time(total_time)}"
        )

        return (
            all_splits,
            processed_results,
            {
                "successful_files": successful_files,
                "total_files": len(pdf_files),
                "total_docs": total_docs,
                "text_docs": text_docs,
                "table_docs": table_docs,
                "total_time": total_time,
            },
        )

    def add_documents(
        self,
        documents: List[Document],
    ):
        self.vectorstore.add_documents(documents)

    def retrieve_documents(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None,
        filter_types: Optional[List[str]] = None,
    ):
        return self.vectorstore.retrieve_documents(
            query=query, filter_sources=filter_sources, filter_types=filter_types
        )

    def clear_vectorstore(self):
        self.vectorstore.clear_vectorstore()

    def get_unique_sources(self):
        return self.vectorstore.get_unique_sources()

    def get_vectorstore(self):
        return self.vectorstore

    def prepare_context(self, docs: List[Document]) -> str:
        context_parts = ["<documents>"]

        for i, doc in enumerate(docs, 1):
            doc_str = f'\n<document index="{i}">'

            doc_str += "\n  <metadata>"
            for key, value in doc.metadata.items():
                doc_str += f"\n    <{key}>{value}</{key}>"
            doc_str += "\n  </metadata>"

            doc_str += f"\n  <content>\n{doc.page_content}\n  </content>"

            doc_str += "\n</document>"

            context_parts.append(doc_str)

        context_parts.append("\n</documents>")

        return "".join(context_parts)


    def __call__(self, query: str, filter: bool = True) -> dict:
        docs = self.retrieve_documents(
            query=query, filter_sources=None, filter_types=None
        )
        context = self.prepare_context(docs)

        return {
            "context": context,
            "docs": docs,
            "query": query,
            "response": ""
        }
