from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


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

    def _safe_text(self, value: Optional[str]) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value).encode("utf-8", errors="replace").decode("utf-8")

    def _extract_figures(self, markdown: str) -> List[Dict[str, str]]:
        figures = []
        for match in re.finditer(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)", markdown or ""):
            alt = match.group("alt").strip() or "Figure"
            url = match.group("url").strip()
            if url:
                figures.append({"alt": alt, "url": url})
        return figures

    def _extract_tables(self, markdown: str) -> List[List[List[str]]]:
        tables = []
        lines = (markdown or "").splitlines()
        i = 0
        while i < len(lines) - 1:
            line = lines[i].strip()
            sep = lines[i + 1].strip()
            if "|" in line and re.match(r"^\s*\|?[\s:-]+\|[\s|:-]+\|?\s*$", sep):
                rows = []
                header = [c.strip() for c in line.strip("|").split("|")]
                rows.append(header)
                i += 2
                while i < len(lines):
                    row_line = lines[i].strip()
                    if not row_line or "|" not in row_line:
                        break
                    row = [c.strip() for c in row_line.strip("|").split("|")]
                    rows.append(row)
                    i += 1
                if len(rows) >= 2:
                    tables.append(rows)
            else:
                i += 1
        return tables

    def _extract_equations(self, markdown: str) -> List[str]:
        equations = []
        if not markdown:
            return equations
        for match in re.finditer(r"\$\$(.+?)\$\$", markdown, flags=re.DOTALL):
            eq = match.group(1).strip()
            if eq:
                equations.append(eq)
        for match in re.finditer(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", markdown):
            eq = match.group(1).strip()
            if eq:
                equations.append(eq)
        return equations

    def _collect_artifacts(self, papers: List) -> Dict[str, List[Dict[str, Any]]]:
        figures = []
        tables = []
        equations = []
        for paper in papers:
            title = getattr(paper.metadata, "title", "Unknown")
            content = paper.full_text or ""
            for fig in self._extract_figures(content):
                figures.append({"paper_title": title, **fig})
            for tbl in self._extract_tables(content):
                tables.append({"paper_title": title, "rows": tbl})
            for eq in self._extract_equations(content):
                equations.append({"paper_title": title, "equation": eq})
        return {"figures": figures, "tables": tables, "equations": equations}
    
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
        """Generate PDF from research results using PyLaTeX."""
        from pylatex import Document, Section, Subsection, Command, Figure, Tabular
        from pylatex.utils import NoEscape, escape_latex

        artifacts = self._collect_artifacts(papers)
        pdf_path = self.exports_dir / f"{query_id}.pdf"

        print(f"[EXPORT] start query_id={query_id}")
        doc = Document(page_numbers=False)
        doc.preamble.append(Command("title", "Research Report"))
        doc.preamble.append(Command("author", "Mini-Perplexity"))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))

        doc.append(NoEscape(r"\noindent\textbf{Query:} " + escape_latex(self._safe_text(query))))
        doc.append(NoEscape(r"\par\textbf{Date:} " + escape_latex(self._safe_text(datetime.now().strftime("%Y-%m-%d %H:%M")))))

        if summary:
            with doc.create(Section("Summary")):
                doc.append(escape_latex(self._safe_text(summary)))

        if artifacts["figures"]:
            with doc.create(Section("Figures")):
                for i, fig in enumerate(artifacts["figures"], 1):
                    with doc.create(Subsection(f"Figure {i}")):
                        with doc.create(Figure(position="h!")) as fig_env:
                            fig_path = fig.get("url", "")
                            if fig_path and Path(fig_path).exists():
                                fig_env.add_image(fig_path, width=NoEscape(r"0.9\textwidth"))
                            caption = f"{fig.get('alt', 'Figure')} (Source: {fig.get('paper_title', 'Unknown')})"
                            fig_env.add_caption(escape_latex(self._safe_text(caption)))
                        if fig.get("url"):
                            doc.append(NoEscape(r"\par\noindent\textbf{Source:} " + escape_latex(self._safe_text(fig["url"]))))

        if artifacts["tables"]:
            with doc.create(Section("Tables")):
                for i, tbl in enumerate(artifacts["tables"], 1):
                    rows = tbl.get("rows") or []
                    if not rows:
                        continue
                    cols = len(rows[0])
                    col_spec = "|".join(["l"] * cols)
                    with doc.create(Subsection(f"Table {i}")):
                        doc.append(NoEscape(r"\noindent\textbf{Source:} " + escape_latex(self._safe_text(tbl.get("paper_title", "Unknown"))) + r"\par"))
                        with doc.create(Tabular("|" + col_spec + "|")) as table:
                            table.add_hline()
                            for r, row in enumerate(rows):
                                row = row + [""] * max(0, cols - len(row))
                                table.add_row([escape_latex(self._safe_text(c)) for c in row[:cols]])
                                if r == 0:
                                    table.add_hline()
                            table.add_hline()

        if artifacts["equations"]:
            with doc.create(Section("Equations")):
                for i, eq in enumerate(artifacts["equations"], 1):
                    with doc.create(Subsection(f"Equation {i}")):
                        doc.append(NoEscape(r"\[" + self._safe_text(eq.get("equation", "")) + r"\]"))
                        doc.append(NoEscape(r"\par\noindent\textbf{Source:} " + escape_latex(self._safe_text(eq.get("paper_title", "Unknown")))))

        doc.generate_pdf(str(pdf_path.with_suffix("")), clean_tex=False)
        print(f"[EXPORT] end query_id={query_id} path={pdf_path}")
        return str(pdf_path)
    
    def get_pdf_path(self, query_id: str) -> Optional[str]:
        """Get path to existing PDF"""
        pdf_path = self.exports_dir / f"{query_id}.pdf"
        return str(pdf_path) if pdf_path.exists() else None
