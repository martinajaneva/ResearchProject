import fitz
import re
from unidecode import unidecode

def text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += " ".join([block[4] for block in page.get_text("blocks")]) + "\n"
    return text

def clean_title(title):
    title = unidecode(title)
    title = title.lower()
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title

extract_conclusions_pattern = re.compile(
    r"(?:^|\n)\s*(?:\d{0,2}[\.]?\s*)?"  
    r"(?P<header>.*?(conclusion|conclusions|discussion|summary|final remarks|closing discussion|broader impact).*?)"  
    r"(?:\s*[:\-]?)\s*[\n ]+" 
    r"(?P<content>.*?)(?=\n\s*(?:\d{1,2}[\. ]+\s*)?[A-Z][^\n]{3,60}\n|\Z)",  
    flags=re.IGNORECASE | re.DOTALL
)

def extract_conclusion_discussion(txt):
    for f in extract_conclusions_pattern.finditer(txt):
        sections = f.group("content").strip()
        if len(sections) > 50:
            return sections
    return None

find_limitations = re.compile(
    r"(?:^|\n)(?:\d{0,2}[\.]\s*)?"
    r"(limitations|limitation|conclusions and limitations|future work|conclusion (?:and|&) future work|limitations (?:and|&) future work|research limitations|study limitations|challenges)"
    r"(?::)?\s*\n+(.*?)(?=\n\s*(?:\d{1,2}[\.]+\s*)?[A-Z][A-Za-z0-9, \-]{3,60}\n|\Z)",
    flags=re.IGNORECASE | re.DOTALL
)

keywords = ["limitations", "future work", "challenges", "limitation", "study limitations", "research limitations", "limitations and future work"]
def find_section(txt):
    paper_sections = {}
    for f in find_limitations.finditer(txt):
        title = f.group(1).strip().lower()
        text = f.group(2).strip()
        if not text.lower().startswith("question: does the paper discuss the limitations"):
            paper_sections[title] = text
    return paper_sections
    
def extract_limitations(txt):
    paper_sections = find_section(txt)
    return paper_sections