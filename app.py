import httpx, pypdf, time, os, json, asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal
from pathlib import Path
from fasthtml.common import *
from dotenv import load_dotenv

load_dotenv()


#------CSS-------------#
minimal_css = Style("""
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        background: #fff;
        color: #000;
        padding: 20px;
        font-size: 15px;
        line-height: 1.6;
    }
    a { color: #b30000; text-decoration: none; border-bottom: 1px solid #0000ee; }
    a:hover { background: #b30000; color: #fff; }
    
    article { border: 2px solid #000; padding: 15px; margin: 15px 0; }
    
    h1, h2 { 
        background: #b30000;
        color: #fff;
        padding: 12px 20px;
        margin: -20px -20px 20px -20px;
    }

    h4 { font-size: 18px; margin-bottom: 8px; }
    form { border: 2px solid #000; padding: 10px; margin-bottom: 15px; }
    label { margin-right: 15px; }
""")



app, rt = fast_app(live=True, pico=False, hdrs=(minimal_css,))

#-----------PROMPTS----------------#
compute_prompt = """Analyze this deep learning paper to assess reproducibility on free Google Colab (Tesla T4: ~15GB VRAM, 12GB RAM, 12hr runtime limit).

Return ONLY valid JSON (no markdown, no code fences):
{{
  "gpu_vram_gb_min": <number or null>,
  "training_time_hours": <number or null>,
  "training_time_confidence": "<stated/estimated/unknown>",
  "multi_gpu_required": <true/false/null>,
  "dataset_publicly_available": <true/false/null>,
  "dataset_size_gb": <number or null>,
  "code_available": <true/false/null>,
  "pretrained_weights_available": <true/false/null>,
  "colab_feasible_rating": <1-5 or null>,
  "colab_feasible_explanation": "<brief explanation>",
  "main_bottleneck": "<vram/compute_time/dataset_access/multi_gpu/no_code/none/unknown>"
}}

Rating scale (1-5):
5: Fully reproducible in <2 hours
4: Reproducible with minor adjustments
3: Possible but requires significant compromises
2: Very difficult, major modifications needed
1: Not reproducible on free Colab

Paper text:
{}"""


improvements_prompt = """Analyze this paper and identify 2-3 improvements suitable for a portfolio project. Prioritize extensions that:

1. Produce clear, demonstrable results you can show in 30 seconds
2. Have a working baseline within 1 week
3. Work on free Google Colab (15GB VRAM, 12hr runtime)
4. Show clear before/after improvements

Return ONLY valid JSON (no markdown, no code fences):
[
  {{
    "improvement": "<specific, achievable experiment>",
    "rationale": "<what skill/insight this demonstrates>",
    "effort": "<low/medium/high>",
    "demo_appeal": "<how impressive/demonstrable the results would be>",
    "category": "<ablation/dataset/comparison/visualization>"
  }}
]

Strong suggestions:
- Apply to a different dataset with clear results
- Add missing comparison to popular baseline
- Create interpretability analysis
- Reproduce with better documentation

Weak suggestions:
- Requires proprietary data or massive compute
- Improvements too subtle to demonstrate clearly
- No clear starting point or existing code

Paper: 
{}"""

#---------------Pydantic & Database Stuff-----------#
class Improvement(BaseModel):
    improvement: str; rationale:str; demo_appeal:str
    effort: Literal['low', 'medium', 'high']
    category: Literal['ablation', 'dataset', 'comparison', 'visualization']

class ComputeRequirements(BaseModel):
    gpu_vram_gb_min: Optional[float] = None
    training_time_hours: Optional[float] = None
    training_time_confidence: Optional[Literal['stated', 'estimated', 'unknown']] = None
    multi_gpu_required: Optional[bool] = None
    dataset_publicly_available: Optional[bool] = None
    dataset_size_gb: Optional[float] = None
    code_available: Optional[bool] = None
    pretrained_weights_available: Optional[bool] = None
    colab_feasible_rating: Optional[int] = Field(None, ge=1, le=5)
    colab_feasible_explanation: Optional[str] = None 
    main_bottleneck: Optional[Literal['vram', 'compute_time', 'dataset_access', 'multi_gpu','no_code', 'none', 'unknown']] = None

@dataclass
class Metadata:
    url:str; text:str; title:str; authors:str; abstract:str; simple_abstract:str; pid:int=None; saved:bool=False;

@dataclass
class Analysis:
    fid:int; pid:int=None
    gpu_vram_gb_min: Optional[float] = None
    training_time_hours: Optional[float] = None
    training_time_confidence: Optional[str] = None
    multi_gpu_required: Optional[bool] = None
    dataset_publicly_available: Optional[bool] = None
    dataset_size_gb: Optional[float] = None
    code_available: Optional[bool] = None
    pretrained_weights_available: Optional[bool] = None
    colab_feasible_rating: Optional[int] = None
    colab_feasible_explanation: Optional[str] = None
    main_bottleneck: Optional[str] = None
    improvements: str = "{}"  # JSON string

    @classmethod
    def from_analysis(cls, fid:int, compute:ComputeRequirements=None, improvements:list[Improvement]=None):
        return cls(fid, improvements=json.dumps([imp.model_dump() for imp in improvements]), **compute.model_dump())


db = database(':memory:')
metadata_t = db.create(Metadata, pk='pid', if_not_exists=True)
analyses_t = db.create(Analysis, pk='pid', foreign_keys=[('fid', 'metadata', 'pid')])


def format_compute(analysis):
    return Div(
        H5("Compute Requirements"),
        Ul(
            Li(f"GPU VRAM: {analysis.gpu_vram_gb_min or 'Unknown'} GB"),
            Li(f"Training Time: {analysis.training_time_hours or 'Unknown'} hours ({analysis.training_time_confidence})"),
            Li(f"Multi-GPU: {'Yes' if analysis.multi_gpu_required else 'No'}"),
            Li(f"Colab Feasible: {analysis.colab_feasible_rating}/5 - {analysis.colab_feasible_explanation}"),
            Li(f"Main Bottleneck: {analysis.main_bottleneck}")
        )
    )

def format_improvements(analysis):
    improvements = json.loads(analysis.improvements)

    items = [
        Li(
            H5(imp['improvement']),
            Small(f"Rationale: {imp['rationale']}"),
            Br(),
            Small(f"Effort: {imp['effort']} | Category: {imp['category']}")
        ) 
        for imp in improvements
    ]

    return Div(
        H5("Suggested Improvements"),
        Ul(*items)
    )
#-----UI-----------------#
def save_btn(pid:int, saved:bool):
    if saved:
        return A('Unsave', hx_post=f'/unsave?pid={pid}', hx_swap='outerHTML')
    return A('Save', hx_post=f'/save?pid={pid}', hx_swap='outerHTML')
    

@rt('/save')
def post(pid:int):
    metadata_t.update({'saved':True}, pid=pid)
    return A('Unsave', hx_post=f'/unsave?pid={pid}', hx_swap='outerHTML')

@rt('/unsave')
def post(pid:int):
    metadata_t.update({'saved':False}, pid=pid)
    return save_btn(pid, False)


@rt('/delete', methods=['delete'])
def delete(pid:int):
    metadata_t.delete(pid)
    if pid in analyses_t:
        analyses_t.delete(pid)
    return ''


def CardFooter(pid:int):
    indicator = f'#loading-{pid}'
    saved = metadata_t[pid].saved
    return Div(
        A('Original', hx_get=f'/original?pid={pid}', hx_target=f'#abstract-text-{pid}'),
        A('Simple', hx_get=f'/simple?pid={pid}', hx_target=f'#abstract-text-{pid}', hx_indicator=indicator),
        A('Compute', hx_get=f'/compute?pid={pid}', hx_target=f'#compute-{pid}', hx_indicator=indicator),
        A('Improvements',hx_get=f'/improvements?pid={pid}', hx_target=f'#improvements-{pid}', hx_indicator=indicator),
        save_btn(pid, saved),
        A('Delete', hx_delete=f'/delete?pid={pid}', hx_target=f'#card-{pid}', hx_swap='delete'),
        style='display:flex; gap:10px;'
    )

def PaperCard(meta:Metadata, pid:int):
    return Card(
        Div(
            H4(
                A(meta.title.title(), href=meta.url)
            ),
            Div('Loading....', id=f'loading-{pid}', cls='htmx-indicator'),
            style='display:flex; gap:10px; align-items:center'
        ),
        P(meta.abstract, id=f'abstract-text-{pid}'),
        Div(id=f'compute-{pid}'), Div(id=f'improvements-{pid}'), CardFooter(pid),
        id=f'card-{pid}',
        style='background-color:#eee; padding:10px; border-radius:5px;'
    )


def AutoCheckbox(label:str, name:str):
    return Label(Input(type='checkbox', name=name, id=name, hx_trigger='change'),label)

filters = Form(hx_get='/filter_papers', hx_trigger='change', hx_target='#papers')(
    AutoCheckbox('Single-GPU', name='gpu1'),
    AutoCheckbox('Public Dataset', name='dataset'),
    AutoCheckbox('Pretrained Weights', name='weights'),
    AutoCheckbox('Colab Feasible >= 3', name='colab_rating'),
    AutoCheckbox('Saved', name='saved'),
    style='display:flex; justify-content:space-between;'
)

#------LLM setup---------#
client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')

simplify_prompt = """Convert this academic abstract to plain English. Remove passive voice, jargon, and overly formal language. Make it clear and engaging for someone who understands CS but dislikes academic writing style.

Output ONLY the plain English version, with no preamble or extra formatting.

Abstract:
{}"""


def run_llm(prompt, content:None):
    """Run any prompt with some optional content"""
    response = client.chat.completions.create(model='deepseek-chat', messages = [{'role':'user', 'content':prompt.format(content)}],)
    return response.choices[0].message.content
    
def simplify_abstract(text): return run_llm(simplify_prompt, text)

def strip_fences(response):
    return response.split('\n', 1)[1].rsplit('\n```',1)[0] if response.startswith('```') else response

def analyze_compute(text):
    response = run_llm(compute_prompt, text)
    return ComputeRequirements.model_validate_json(strip_fences(response))

def analyze_improvements(text):
    response = run_llm(improvements_prompt, text)
    return [Improvement.model_validate(imp) for imp in json.loads(strip_fences(response))]

@rt('/original')
def get(pid:int): return metadata_t[pid].abstract

@rt('/simple')
async def get(pid:int):
    simple = metadata_t[pid].simple_abstract
    if simple is None:
        simple = await asyncio.to_thread(simplify_abstract, metadata_t[pid].abstract)
        meta = metadata_t.update({'simple_abstract':simple}, pid=pid)
    return simple

# are compute filds empty
def empty_compute(analysis):
    return all(getattr(analysis, k) is None for k in ComputeRequirements.model_fields.keys())

def load_or_create_analysis(pid:int):
    results = analyses_t(where='fid=?', where_args=[pid])
    if results:
        analysis = results[0]
    else:
        analysis = analyses_t.insert(Analysis(fid=pid))
    has_compute = not empty_compute(analysis)
    has_improvements = analysis.improvements not in ["{}", None]
    return analysis, has_compute, has_improvements

@rt('/compute')
async def get(pid:int):
    analysis, has_compute, has_improvements = load_or_create_analysis(pid)
    paper = metadata_t[pid]
    if not has_compute:
        compute = await asyncio.to_thread(analyze_compute, paper.text)
        for k, v in compute.model_dump().items():
            setattr(analysis, k, v)
        analysis = analyses_t.update(analysis)
    return Div(format_compute(analysis),
               A('remove', hx_get=f'/remove/compute/{pid}', hx_swap='outerHTML', hx_target=f'#compute-{pid}'),
               id=f'compute-{pid}')

@rt('/improvements')
async def get(pid:int):
    analysis, has_compute, has_improvements = load_or_create_analysis(pid)
    paper = metadata_t[pid]
    if not has_improvements:
        improvements = await asyncio.to_thread(analyze_improvements, paper.text)
        analysis.improvements = json.dumps([imp.model_dump() for imp in improvements])
        analysis = analyses_t.update(analysis)
    return Div(format_improvements(analysis), 
               A('remove', hx_get=f'/remove/improvements/{pid}', hx_swap='outerHTML', hx_target=f'#improvements-{pid}'), id=f'improvements-{pid}')

@rt('/remove/{section}/{pid}')
def remove(section:str, pid:int): return Div(id=f'{section}-{pid}')

limit = 10
more_link = A('Load More..', 
              hx_get='/load_more', 
              hx_swap='beforeend', 
              hx_target='#papers', 
              hx_vals='js:{count: document.querySelectorAll("#papers > *").length}',
              id='more-link') # sends count of loaded cards

@rt('/load_more')
def load_more(count:int, sess):
    papers = metadata_t(limit=limit, offset=count)
    cards = [PaperCard(p, p.pid) for p in papers]
    if count + limit >= len(metadata_t()):
        return (*cards, Span(id='more-link', hx_swap_oob='true'))
    return (*cards, )


@rt('/filter_papers')
def get(gpu1:bool=False, dataset:bool=False, weights:bool=False, colab_rating:int=None, saved:bool=None):
    conditions = []
    if gpu1: conditions.append("multi_gpu_required=0")
    if dataset: conditions.append("dataset_publicly_available=1")
    if weights: conditions.append("pretrained_weights_available=1")
    if colab_rating: conditions.append("colab_feasible_rating >= 3")

    if conditions or saved:
        if conditions:
            rows = analyses_t(where=' AND '.join(conditions))
            ids = [p.fid for p in rows]
            placeholders = ','.join('?' * len(ids))
    
            metadata_conditions = [f"pid IN ({placeholders})"]
            metadata_args = list(ids)
        else:
            # No analysis filters, so don't restrict by pid
            metadata_conditions = []
            metadata_args = []

        if saved:
            metadata_conditions.append("saved=1")

        where_clause = ' AND '.join(metadata_conditions) if metadata_conditions else None
        papers = metadata_t(where=where_clause, where_args=metadata_args or None)
        cards = [PaperCard(p, p.pid) for p in papers]
        return Div(*cards, id='papers')
    else:
        return Redirect('/')


#-----------GET NEW PAPERS------------#
def get_pages(r:httpx.Response) -> list[str]:
    """Returns the extracted text content of all PDF pages as a list."""
    reader = pypdf.PdfReader(BytesIO(r.content))
    pages = [page.extract_text() for page in reader.pages]
    return pages

def paper_metadata(pages:list[str]) -> dict[str]:
    """Applies heuristics to raw text to extract title, authors, and abstract."""
    first_page = pages[0].lower()
    abstract_ix, intro_ix = first_page.find('abstract'), first_page.find('introduction')
    if abstract_ix == -1 or intro_ix == -1:
        return {"title": "Unknown Title", "authors": "Unknown Authors", "abstract": "Extraction failed."}

    abstract = first_page[abstract_ix+8:intro_ix].strip() # +8 is len('abstract')
    page_header_text = first_page[:abstract_ix].split('\n')
    page_header_lines = [i for i in page_header_text if i.strip() != '']
    title = page_header_lines[0]
    authors = ', '.join(page_header_lines[1:]) if len(page_header_lines[1:]) > 0 else "Unknown Authors"
    text = '\n'.join(pages)
    return {'title':title, 'authors':authors, 'abstract':abstract, 'text':text}


paper_fetch_form = Form(hx_get='/fetch_paper', hx_target='#papers', hx_swap='afterbegin', style='display:flex; gap:10px;')(
    Input(type='url', name='url', placeholder='pdf link..'),
    Button('Fetch'),
    Div(id='error-msg'))

@rt('/fetch_paper')
def get(url:str):
    r = httpx.get(url)
    if r.status_code == 200:
        pages = get_pages(r)
        meta = paper_metadata(pages)
        paper_meta = Metadata(url=url, **meta)
        existing = db.q('select * from metadata where url=?', (paper_meta.url,))
        if not existing:
            p = metadata_t.insert(paper_meta)
            return PaperCard(p, p.pid)
        else:
            return Div('Paper already exists', id='error-msg', hx_swap_oob='true', hx_swap='delete', hx_trigger='load delay:3s', style='color:red;')

    else:
        return Div(f'Could not fetch paper {r.status_code}', id='error-msg', hx_swap='delete', hx_trigger='load delay:3s', style='color:red;')

@rt('/')
def index(sess):
    papers = metadata_t(limit=limit, offset=0)
    cards = [PaperCard(p, p.pid) for p in papers]
    return Titled('paper-sanity', paper_fetch_form, filters, Div(*cards, id='papers'), more_link)

serve()