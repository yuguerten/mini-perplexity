from pathlib import Path
from typing import List, Optional
from datetime import datetime

from weasyprint import HTML, CSS


class ExportService:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.exports_dir = self.storage_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_markdown(
        self,
        query: str,
        papers: List,
        summary: Optional[str] = None
    ) -> str:
        """Generate markdown content for the research"""
        md = []
        
        md.append(f"# Research Report")
        md.append(f"\n**Query:** {query}")
        md.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        if summary:
            md.append("\n---\n")
            md.append("## Summary")
            md.append(f"\n{summary}")
        
        md.append("\n---\n")
        md.append("## Papers")
        
        for i, paper in enumerate(papers, 1):
            metadata = paper.metadata
            md.append(f"\n### {i}. {metadata.title}")
            md.append(f"\n**Authors:** {', '.join(metadata.authors) if metadata.authors else 'Unknown'}")
            
            if metadata.published_date:
                md.append(f"\n**Published:** {metadata.published_date}")
            
            if metadata.citations:
                md.append(f"\n**Citations:** {metadata.citations}")
            
            md.append(f"\n**Source:** {metadata.source}")
            md.append(f"\n**URL:** {metadata.url}")
            
            if metadata.abstract:
                md.append(f"\n**Abstract:** {metadata.abstract}")
            
            if paper.summary:
                md.append(f"\n**Summary:** {paper.summary}")
            
            md.append("\n")
        
        md.append("\n---\n")
        md.append("## References")
        
        for i, paper in enumerate(papers, 1):
            metadata = paper.metadata
            ref = f"\n[{i}] {metadata.title}. "
            if metadata.authors:
                ref += f"{', '.join(metadata.authors[:3])}"
                if len(metadata.authors) > 3:
                    ref += " et al."
            if metadata.published_date:
                ref += f". ({metadata.published_date[:4] if len(metadata.published_date) >= 4 else metadata.published_date})"
            ref += f". {metadata.source}. {metadata.url}"
            md.append(ref)
        
        return "\n".join(md)
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML for PDF generation"""
        import re
        
        html = markdown
        
        html = html.replace("&", "&amp;")
        html = html.replace("<", "&lt;")
        html = html.replace(">", "&gt;")
        
        lines = html.split("\n")
        result = []
        
        in_code_block = False
        in_list = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("```"):
                if in_code_block:
                    in_code_block = False
                    result.append("</code></pre>")
                else:
                    in_code_block = True
                    result.append("<pre><code>")
                continue
            
            if in_code_block:
                result.append(line)
                continue
            
            if line.startswith("# "):
                result.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                result.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                result.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("**") and "**" in line[2:]:
                line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
                result.append(f"<p>{line}</p>")
            elif line.startswith("- ") or line.startswith("* "):
                if not in_list:
                    result.append("<ul>")
                    in_list = True
                result.append(f"<li>{line[2:]}</li>")
            elif line.startswith("[") and "](" in line:
                line = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r'<a href="\2">\1</a>', line)
                result.append(f"<p>{line}</p>")
            elif line == "---":
                result.append("<hr/>")
            elif line:
                result.append(f"<p>{line}</p>")
            else:
                if in_list:
                    result.append("</ul>")
                    in_list = False
        
        if in_list:
            result.append("</ul>")
        
        return "\n".join(result)
    
    def generate_pdf(
        self,
        query_id: str,
        query: str,
        papers: List,
        summary: Optional[str] = None
    ) -> str:
        """Generate PDF from research results"""
        markdown = self._generate_markdown(query, papers, summary)
        html_content = self._markdown_to_html(markdown)
        
        html_document = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Research Report - {query}</title>
            <style>
                @page {{
                    margin: 2cm;
                    size: A4;
                }}
                body {{
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    font-size: 11pt;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    color: #1a1a1a;
                    border-bottom: 2px solid #0066cc;
                    padding-bottom: 10px;
                    font-size: 24pt;
                }}
                h2 {{
                    color: #333;
                    margin-top: 30px;
                    font-size: 16pt;
                }}
                h3 {{
                    color: #555;
                    font-size: 13pt;
                    margin-top: 20px;
                }}
                p {{
                    margin: 10px 0;
                    text-align: justify;
                }}
                ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                li {{
                    margin: 5px 0;
                }}
                a {{
                    color: #0066cc;
                    text-decoration: none;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #ccc;
                    margin: 30px 0;
                }}
                strong {{
                    color: #1a1a1a;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        pdf_path = self.exports_dir / f"{query_id}.pdf"
        
        HTML(string=html_document).write_pdf(str(pdf_path))
        
        return str(pdf_path)
    
    def get_pdf_path(self, query_id: str) -> Optional[str]:
        """Get path to existing PDF"""
        pdf_path = self.exports_dir / f"{query_id}.pdf"
        return str(pdf_path) if pdf_path.exists() else None
