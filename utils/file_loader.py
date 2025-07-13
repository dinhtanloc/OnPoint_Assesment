from loguru import logger
import os
import tempfile
from typing import List

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from .setting import CHUNK_OVERLAP, CHUNK_SIZE
from .parser import SimpleParser


class TextSplitter:
    def __init__(
        self,
        chunk_type: str = "recursive",
        separators: List[str] = ["\n\n", "\n", ". ", "! ", "? ", ":", ";", " "],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separator: str = "\n\n",
    ) -> None:
        self.chunk_type = chunk_type
        
        if chunk_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
        elif chunk_type == "character":
            self.splitter = CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
        else:
            raise ValueError(f"Unsupported chunk_type: {chunk_type}. Use 'recursive' or 'character'.")

    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class PDFLoader:
    def __init__(self, text_splitter=None, debug: bool = False, temp_dir: str = None):
        """
        Initialize the PDF loader with an optional text splitter and credentials for WDMPDFParser.

        Args:
            text_splitter: An instance of TextSplitter to process text documents
            debug: Enable debug mode for detailed logging
            temp_dir: Custom temporary directory path (optional)
        """
        self.text_splitter = text_splitter or TextSplitter()
        self.debug = debug
        self.temp_dir = temp_dir
        # Create temp directory if specified and doesn't exist
        if self.temp_dir and not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
            if self.debug:
                logger.info(f"Created temporary directory: {self.temp_dir}")

    def _load_from_file_object(self, pdf_file, original_filename: str = None):
        """
        Helper method to load a PDF from a file object using WDMPDFParser.

        Args:
            pdf_file: A file object from Streamlit's file_uploader
            original_filename: Original filename for metadata

        Returns:
            List[Document]: Documents from the PDF (text and tables)
        """
        if self.debug:
            logger.info(f"Processing PDF file: {original_filename or 'unnamed'}")
            logger.info(f"File size: {len(pdf_file.getvalue())} bytes")
        
        temp_file_kwargs = {"delete": False, "suffix": ".pdf"}
        if self.temp_dir:
            temp_file_kwargs["dir"] = self.temp_dir
            
        with tempfile.NamedTemporaryFile(**temp_file_kwargs) as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_path = tmp_file.name
            
        if self.debug:
            logger.info(f"Created temporary file: {temp_path}")

        try:
            # Initialize WDMPDFParser
            parser = SimpleParser(
                file_path=temp_path,
                debug=self.debug  # Pass debug flag to parser
            )
            
            if self.debug:
                logger.info("Extracting text documents...")
            
            # Extract text documents
            text_documents = parser.extract_text()
            
            if self.debug:
                logger.info(f"Extracted {len(text_documents)} text documents")
                logger.info("Extracting table documents...")
            
            # Extract table documents (only if credentials are available)
            table_documents = []
            try:
                if self.debug:
                    logger.info("Using basic mode (without credentials)")
                table_documents = parser.extract_tables(
                    merge_span_tables=False,
                )
                logger.info(f"Extracted {len(table_documents)} tables from PDF (basic mode)")
            except Exception as e:
                logger.warning(f"Failed to extract tables from {original_filename or 'unnamed'}: {e}")
                if self.debug:
                    logger.exception("Table extraction error details:")
                table_documents = []
            
            # Combine all documents
            all_documents = text_documents + table_documents
            
            # Update metadata with original filename if provided
            if original_filename:
                for doc in all_documents:
                    doc.metadata["source"] = original_filename
            
            if self.debug:
                logger.info(f"Total documents extracted from {original_filename or 'unnamed'}: {len(all_documents)} (Text: {len(text_documents)}, Tables: {len(table_documents)})")
                
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to process PDF {original_filename or 'unnamed'}: {e}")
            if self.debug:
                logger.exception("PDF processing error details:")
            return []
            
        finally:
            # Ensure cleanup happens even if loading fails
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    if self.debug:
                        logger.info(f"Cleaned up temporary file: {temp_path}")
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")

    def _load_from_path(self, path_string):
        """
        Helper method to load a PDF from a file path using WDMPDFParser.

        Args:
            path_string: A string path to a PDF file

        Returns:
            List[Document]: Documents from the PDF (text and tables)
        """
        if self.debug:
            logger.info(f"Processing PDF from path: {path_string}")
            
        # Check if file exists
        if not os.path.exists(path_string):
            logger.error(f"PDF file not found: {path_string}")
            return []
            
        try:
            # Initialize WDMPDFParser
            parser = SimpleParser(
                file_path=path_string,
                debug=self.debug  # Pass debug flag to parser
            )
            
            if self.debug:
                logger.info("Extracting text documents...")
            
            # Extract text documents
            text_documents = parser.extract_text()
            
            if self.debug:
                logger.info(f"Extracted {len(text_documents)} text documents")
                logger.info("Extracting table documents...")
            
            table_documents = []
            try:
                if self.debug:
                    logger.info("Using basic mode (without credentials)")
                table_documents = parser.extract_tables(
                    merge_span_tables=False,
                )
                logger.info(f"Extracted {len(table_documents)} tables from PDF (basic mode)")
            except Exception as e:
                logger.warning(f"Failed to extract tables from {path_string}: {e}")
                if self.debug:
                    logger.exception("Table extraction error details:")
                table_documents = []
            
            all_documents = text_documents + table_documents
            
            if self.debug:
                logger.info(f"Total documents extracted from {path_string}: {len(all_documents)} (Text: {len(text_documents)}, Tables: {len(table_documents)})")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to process PDF {path_string}: {e}")
            if self.debug:
                logger.exception("PDF processing error details:")
            return []

    def load(self, pdf_file=None, path_string=None, original_filename: str = None):
        """
        Load PDF content either from a file uploaded via Streamlit or from a path.

        Args:
            pdf_file: A file object from Streamlit's file_uploader
            path_string: A string path to a PDF file
            original_filename: Original filename for metadata

        Returns:
            List[Document]: A list of document chunks after splitting (only text is split, tables remain whole)

        Raises:
            ValueError: If neither pdf_file nor path_string is provided
        """
        if pdf_file is not None:
            documents = self._load_from_file_object(pdf_file, original_filename)
        elif path_string is not None:
            documents = self._load_from_path(path_string)
        else:
            raise ValueError("Either pdf_file or path_string must be provided")

        if not documents:
            return []

        text_documents = [doc for doc in documents if doc.metadata.get("type") == "text"]
        table_documents = [doc for doc in documents if doc.metadata.get("type") == "table"]
        
        split_text_documents = self.text_splitter(text_documents) if text_documents else []
        
        logger.info(f"Text documents split into {len(split_text_documents)} chunks")
        logger.info(f"Table documents kept whole: {len(table_documents)} tables")
        
        return split_text_documents + table_documents

