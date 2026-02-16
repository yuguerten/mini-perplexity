import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings


class ChunkedDocument:
    def __init__(self, paper_id: str, section: str, content: str, embedding: Optional[List[float]] = None):
        self.paper_id = paper_id
        self.section = section
        self.content = content
        self.embedding = embedding
        self.id = f"{paper_id}_{section}_{uuid.uuid4().hex[:8]}"


class RAGService:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.chroma_dir = self.storage_dir / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"description": "Academic papers for RAG"}
        )
    
    def chunk_by_section(self, paper_id: str, content: str) -> List[ChunkedDocument]:
        """Chunk document by academic sections"""
        chunks = []
        
        section_patterns = {
            "abstract": r"(?i)^#?\s*(?:abstract|summary)\s*$",
            "introduction": r"(?i)^#?\s*(?:introduction|background)\s*$",
            "methods": r"(?i)^#?\s*(?:methods?|methodology|approach)\s*$",
            "results": r"(?i)^#?\s*(?:results?|findings)\s*$",
            "discussion": r"(?i)^#?\s*(?:discussion|analysis)\s*$",
            "conclusion": r"(?i)^#?\s*(?:conclusion|conclusions|future.?work)\s*$",
            "references": r"(?i)^#?\s*(?:references|bibliography|works.?cited)\s*$",
        }
        
        import re
        
        lines = content.split("\n")
        current_section = "general"
        current_content = []
        
        for line in lines:
            is_header = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line.strip()):
                    if current_content:
                        chunk_text = "\n".join(current_content).strip()
                        if chunk_text:
                            chunks.append(ChunkedDocument(
                                paper_id=paper_id,
                                section=current_section,
                                content=chunk_text
                            ))
                    current_section = section_name
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line)
        
        if current_content:
            chunk_text = "\n".join(current_content).strip()
            if chunk_text:
                chunks.append(ChunkedDocument(
                    paper_id=paper_id,
                    section=current_section,
                    content=chunk_text
                ))
        
        if not chunks:
            chunk_size = 2000
            for i in range(0, len(content), chunk_size):
                chunks.append(ChunkedDocument(
                    paper_id=paper_id,
                    section=f"part_{i // chunk_size + 1}",
                    content=content[i:i + chunk_size]
                ))
        
        return chunks
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        try:
            import ollama
            
            embeddings = []
            for text in texts:
                response = ollama.embeddings(
                    model="nomic-embed-text",
                    prompt=text[:8000]
                )
                embeddings.append(response.get("embedding", []))
            
            return embeddings
        except Exception as e:
            print(f"Embedding error: {e}")
            return [[0.0] * 768 for _ in texts]
    
    def add_paper(self, paper_id: str, content: str, metadata: Dict[str, Any]):
        """Add paper to vector store"""
        chunks = self.chunk_by_section(paper_id, content)
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "paper_id": chunk.paper_id,
                    "section": chunk.section,
                    **metadata
                }
                for chunk in chunks
            ]
        )
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query the vector store"""
        query_embedding = self._get_embeddings([query_text])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        sources = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                sources.append({
                    "content": doc,
                    "paper_id": results["metadatas"][0][i].get("paper_id"),
                    "section": results["metadatas"][0][i].get("section"),
                    "title": results["metadatas"][0][i].get("title"),
                    "distance": results["distances"][0][i] if "distances" in results else None,
                })
        
        return sources
    
    def generate_answer(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Generate answer using Ollama with RAG context"""
        try:
            import ollama
            
            context = "\n\n".join([
                f"[{i+1}] {s.get('title', 'Unknown')} - {s.get('section', 'General')}:\n{s.get('content', '')}"
                for i, s in enumerate(sources)
            ])
            
            system_prompt = """You are an academic research assistant.
Answer the user's question based on the provided context from academic papers.
Be precise, cite sources using [n] format, and maintain academic tone."""
            
            user_prompt = f"""Question: {query}

Context from papers:

{context}

Provide a detailed answer citing sources [1], [2], etc."""
            
            response = ollama.chat(
                model="deepseek-r1:1.5b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            return response.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"Answer generation error: {e}")
            return "Error generating answer. Please try again."
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total_documents": self.collection.count(),
            "papers": self.collection.get(include=["metadatas"])["metadatas"]
        }
    
    def delete_paper(self, paper_id: str):
        """Delete paper from vector store"""
        results = self.collection.get(where={"paper_id": paper_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
    
    def clear_all(self):
        """Clear all documents from collection"""
        self.collection.delete(where={})
