import json
import logging
import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.settings import settings

logger = logging.getLogger(__name__)

class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report_hierarchical(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """
        Split report into hierarchical chunks with parent/child relationships and typed chunks.
        """
        all_chunks = []
        chunk_id_counter = 0

        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))

        for page in file_content['content']['pages']:
            # Create hierarchical text chunks
            hierarchical_chunks = self._split_page_hierarchical(page)

            # Add parents
            for parent in hierarchical_chunks['parents']:
                parent['id'] = chunk_id_counter
                chunk_id_counter += 1
                all_chunks.append(parent)

            # Add children
            for child in hierarchical_chunks['children']:
                child['id'] = chunk_id_counter
                chunk_id_counter += 1
                all_chunks.append(child)

            # Add table chunks
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table_chunk = {
                        "id": chunk_id_counter,
                        "type": "table",
                        "page_from": page['page'],
                        "page_to": page['page'],
                        "text": table['text'],
                        "table_id": table.get('table_id'),
                        "length_tokens": table['length_tokens']
                    }
                    all_chunks.append(table_chunk)
                    chunk_id_counter += 1

        file_content['content']['chunks'] = all_chunks
        return file_content

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split report into chunks, preserving markdown tables in content and optionally including serialized tables."""
        chunks = []
        chunk_id = 0

        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))

        for page in file_content['content']['pages']:
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)

            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)

        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count

    def _split_page_hierarchical(self, page: Dict[str, any], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> Dict[str, List[Dict[str, any]]]:
        """
        Split page into hierarchical chunks: parent chunks and child chunks.

        Returns:
            Dict with 'parents' and 'children' lists
        """
        page_text = page['text']
        page_num = page['page']

        if chunk_size is None:
            chunk_size = settings.chunk_size_tokens
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap_tokens

        # Create parent chunks (larger context units)
        parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size * 3,  # Larger parent chunks
            chunk_overlap=chunk_overlap
        )

        parent_texts = parent_splitter.split_text(page_text)
        parents = []
        children = []

        parent_id_counter = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_id = f"parent_{page_num}_{parent_idx}"

            # Create parent chunk
            parent_chunk = {
                "id": parent_id,
                "type": "parent",
                "page_from": page_num,
                "page_to": page_num,
                "text": parent_text,
                "length_tokens": self.count_tokens(parent_text)
            }
            parents.append(parent_chunk)

            # Create child chunks from parent
            child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4o",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            child_texts = child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_texts):
                child_chunk = {
                    "id": f"child_{page_num}_{parent_idx}_{child_idx}",
                    "type": "child",
                    "parent_id": parent_id,
                    "page_from": page_num,
                    "page_to": page_num,
                    "text": child_text,
                    "length_tokens": self.count_tokens(child_text)
                }
                children.append(child_chunk)

        return {"parents": parents, "children": children}

    def _split_page(self, page: Dict[str, any], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[Dict[str, any]]:
        """Split page text into chunks. The original text includes markdown tables."""
        if chunk_size is None:
            chunk_size = settings.chunk_size_tokens
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap_tokens
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    def _dedupe_chunks(self, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        if not settings.chunk_dedup:
            return chunks
        seen = set()
        deduped = []
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            norm = re.sub(r"\s+", " ", text).lower()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(chunk)
        if len(deduped) != len(chunks):
            logger.info("Chunk dedup: %d -> %d", len(chunks), len(deduped))
        return deduped

    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):

        all_report_paths = list(all_report_dir.glob("*.json"))
        
        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")
                
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            updated_report = self._split_report(report_data, serialized_tables_path)
            updated_report["content"]["chunks"] = self._dedupe_chunks(updated_report["content"]["chunks"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        print(f"Split {len(all_report_paths)} files")
