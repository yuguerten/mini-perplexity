from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import re
import subprocess
import unicodedata
import os


class ExportService:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.exports_dir = self.storage_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.max_figures = int(os.getenv("REPORT_MAX_FIGURES", "8"))
        self.max_tables = int(os.getenv("REPORT_MAX_TABLES", "8"))
        self.max_equations = int(os.getenv("REPORT_MAX_EQUATIONS", "12"))
    
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
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value).encode("utf-8", errors="replace").decode("utf-8")

        # Normalize Unicode and replace spacing/control chars that break pdflatex.
        text = unicodedata.normalize("NFKC", text)
        text = text.translate({
            0x00A0: " ",  # non-breaking space
            0x202F: " ",  # narrow non-breaking space
            0x2007: " ",  # figure space
            0x2009: " ",  # thin space
            0x200A: " ",  # hair space
            0x2060: "",   # word joiner
            0xFEFF: "",   # zero-width no-break space
            0x2713: "yes",   # check mark
            0x2714: "yes",   # heavy check mark
            0x2717: "no",    # ballot x
            0x2718: "no",    # heavy ballot x
            0x274C: "no",    # cross mark
            0x2705: "yes",   # white heavy check mark
            0xFE0F: "",      # variation selector-16
        })
        return text

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

    def _build_reference_text(self, paper: Any) -> str:
        metadata = paper.metadata
        authors = metadata.authors or []
        if len(authors) > 4:
            author_text = ", ".join(authors[:4]) + ", et al."
        else:
            author_text = ", ".join(authors) if authors else "Unknown authors"

        year = ""
        if metadata.published_date:
            year = self._safe_text(metadata.published_date)[:4]

        title = self._safe_text(metadata.title or "Untitled")
        source = self._safe_text(metadata.source or "Unknown source")
        url = self._safe_text(metadata.url or "")

        parts = [author_text, f'"{title}"']
        if year:
            parts.append(f"({year})")
        parts.append(source)
        if url:
            parts.append(url)
        return ". ".join([p for p in parts if p])

    def _inline_to_latex(self, text: str) -> str:
        from pylatex.utils import escape_latex

        line = self._safe_text(text).strip()
        if not line:
            return ""

        # Convert markdown links to plain text with URL.
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        # Strip markdown emphasis markers to avoid raw tokens in PDF.
        line = line.replace("**", "").replace("__", "").replace("*", "")

        math_chunks: Dict[str, str] = {}

        def hold_math(match: re.Match[str]) -> str:
            key = f"MATHPH{len(math_chunks)}"
            math_chunks[key] = match.group(0)
            return key

        line = re.sub(r"\\\(.+?\\\)", hold_math, line)
        line = re.sub(r"\\\[(?:.|\n)+?\\\]", hold_math, line)

        escaped = escape_latex(line)
        for key, value in math_chunks.items():
            escaped = escaped.replace(key, value)
        return escaped

    def _append_summary_content(self, doc: Any, summary: str) -> None:
        from pylatex.utils import NoEscape

        lines = self._safe_text(summary).splitlines()
        in_itemize = False
        in_enumerate = False

        def close_lists():
            nonlocal in_itemize, in_enumerate
            if in_itemize:
                doc.append(NoEscape(r"\end{itemize}"))
                in_itemize = False
            if in_enumerate:
                doc.append(NoEscape(r"\end{enumerate}"))
                in_enumerate = False

        for raw in lines:
            line = raw.strip()
            if not line or line == "---":
                close_lists()
                continue
            if re.match(r"^\|?[-:\s|]+\|[-:\s|]*$", line):
                continue
            if line.startswith("# "):
                close_lists()
                doc.append(NoEscape(r"\section*{" + self._inline_to_latex(line[2:]) + "}"))
                continue
            if line.startswith("## "):
                close_lists()
                doc.append(NoEscape(r"\subsection*{" + self._inline_to_latex(line[3:]) + "}"))
                continue
            if line.startswith("### "):
                close_lists()
                doc.append(NoEscape(r"\subsubsection*{" + self._inline_to_latex(line[4:]) + "}"))
                continue
            num_match = re.match(r"^\d+\.\s+(.*)$", line)
            if num_match:
                if in_itemize:
                    doc.append(NoEscape(r"\end{itemize}"))
                    in_itemize = False
                if not in_enumerate:
                    doc.append(NoEscape(r"\begin{enumerate}"))
                    in_enumerate = True
                doc.append(NoEscape(r"\item " + self._inline_to_latex(num_match.group(1))))
                continue
            if line.startswith("- ") or line.startswith("* "):
                if in_enumerate:
                    doc.append(NoEscape(r"\end{enumerate}"))
                    in_enumerate = False
                if not in_itemize:
                    doc.append(NoEscape(r"\begin{itemize}"))
                    in_itemize = True
                doc.append(NoEscape(r"\item " + self._inline_to_latex(line[2:])))
                continue
            close_lists()
            doc.append(NoEscape(self._inline_to_latex(line) + r"\par"))

        close_lists()
    
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
        doc = Document(page_numbers=False, geometry_options={"margin": "1in"})
        doc.preamble.append(Command("title", "Research Report"))
        doc.preamble.append(Command("author", "Mini-Perplexity"))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.preamble.append(NoEscape(r"\sloppy"))
        doc.append(NoEscape(r"\maketitle"))

        doc.append(NoEscape(r"\noindent\textbf{Query:} " + escape_latex(self._safe_text(query))))
        doc.append(NoEscape(r"\par\textbf{Date:} " + escape_latex(self._safe_text(datetime.now().strftime("%Y-%m-%d %H:%M")))))

        if summary:
            doc.append(NoEscape(r"\section*{Summary}"))
            self._append_summary_content(doc, summary)

        figures = artifacts["figures"][: max(0, self.max_figures)]
        tables = artifacts["tables"][: max(0, self.max_tables)]
        equations = artifacts["equations"][: max(0, self.max_equations)]

        if figures:
            with doc.create(Section("Figures")):
                for i, fig in enumerate(figures, 1):
                    with doc.create(Subsection(f"Figure {i}")):
                        with doc.create(Figure(position="h!")) as fig_env:
                            fig_path = fig.get("url", "")
                            if fig_path and Path(fig_path).exists():
                                fig_env.add_image(fig_path, width=NoEscape(r"0.9\textwidth"))
                            caption = f"{fig.get('alt', 'Figure')} (Source: {fig.get('paper_title', 'Unknown')})"
                            fig_env.add_caption(escape_latex(self._safe_text(caption)))
                        if fig.get("url"):
                            doc.append(NoEscape(r"\par\noindent\textbf{Source:} " + escape_latex(self._safe_text(fig["url"]))))

        if tables:
            with doc.create(Section("Tables")):
                for i, tbl in enumerate(tables, 1):
                    rows = tbl.get("rows") or []
                    if not rows:
                        continue
                    if len(rows[0]) > 8 or len(rows) > 30:
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

        if equations:
            with doc.create(Section("Equations")):
                for i, eq in enumerate(equations, 1):
                    with doc.create(Subsection(f"Equation {i}")):
                        doc.append(NoEscape(r"\[" + self._safe_text(eq.get("equation", "")) + r"\]"))
                        doc.append(NoEscape(r"\par\noindent\textbf{Source:} " + escape_latex(self._safe_text(eq.get("paper_title", "Unknown")))))

        if papers:
            doc.append(NoEscape(r"\section*{References}"))
            doc.append(NoEscape(r"\begin{enumerate}"))
            for paper in papers:
                ref_text = self._build_reference_text(paper)
                doc.append(NoEscape(r"\item " + self._inline_to_latex(ref_text)))
            doc.append(NoEscape(r"\end{enumerate}"))

        tex_base = pdf_path.with_suffix("")
        doc.generate_tex(str(tex_base))

        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"{tex_base.name}.tex",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(self.exports_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        if proc.returncode != 0:
            stderr_text = proc.stderr.decode("utf-8", errors="replace")
            stdout_tail = proc.stdout[-2000:].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"pdflatex failed rc={proc.returncode}. stderr={stderr_text.strip()} stdout_tail={stdout_tail.strip()}"
            )

        print(f"[EXPORT] end query_id={query_id} path={pdf_path}")
        return str(pdf_path)
    
    def get_pdf_path(self, query_id: str) -> Optional[str]:
        """Get path to existing PDF"""
        pdf_path = self.exports_dir / f"{query_id}.pdf"
        return str(pdf_path) if pdf_path.exists() else None
