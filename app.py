from fasthtml.common import *
import json
from pydantic import BaseModel, Field 
from typing import Optional, Literal


app, rt = fast_app(live=True)

#---------------Database Stuff-----------#



class Improvement(BaseModel):
    improvement: str; rationale:str; demo_appeal:str
    effort: Literal['low', 'medium', 'high']
    category: Literal['ablation', 'dataset', 'comparison', 'visualization']

class ComputeRequirements(BaseModel):
    gpu_vram_gb_min: Optional[float] = None
    training_time_hours: Optional[float] = None
    training_time_confidence: Literal['stated', 'estimated', 'unknown']
    multi_gpu_required: Optional[bool] = None
    dataset_publicly_available: Optional[bool] = None
    dataset_size_gb: Optional[float] = None
    code_available: Optional[bool] = None
    pretrained_weights_available: Optional[bool] = None
    colab_feasible_rating: Optional[int] = Field(None, ge=1, le=5)
    colab_feasible_explanation: str
    main_bottleneck: Literal['vram', 'compute_time', 'dataset_access', 'multi_gpu','no_code', 'none', 'unknown']


@dataclass
class Metadata:
    url:str; text:str; title:str; authors:str; abstract:str; simple_abstract:str; pid:int=None;

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
    def from_analysis(cls, fid:int, compute:ComputeRequirements, improvements=list[Improvement]):
        return cls(fid, improvements=json.dumps([imp.model_dump() for imp in improvements]), **compute.model_dump())


db = database(':memory:')
metadata_t = db.create(Metadata, pk='pid', if_not_exists=True)
analyses_t = db.create(Analysis, pk='pid', foreign_keys=[('fid', 'metadata', 'pid')])

# Insert a few sample papers
metadata_t.insert(
    url='https://arxiv.org/abs/2301.00001',
    text='Full paper text here...',
    title='Attention Is All You Need v2',
    authors='Smith, J. and Jones, M.',
    abstract='We propose a new transformer architecture that...',
    simple_abstract='A new way to make AI models faster.'
)

metadata_t.insert(
    url='https://arxiv.org/abs/2301.00002',
    text='Another paper...',
    title='Efficient Training of Large Models',
    authors='Chen, L. et al.',
    abstract='Training large neural networks requires...',
    simple_abstract='Tips for training big AI models cheaply.'
)


# For paper 1 (Transformer paper)
analyses_t.insert(
    fid=1, pid=1,
    gpu_vram_gb_min=16.0,
    training_time_hours=48.0,
    training_time_confidence='estimated',
    multi_gpu_required=False,
    dataset_publicly_available=True,
    dataset_size_gb=5.2,
    code_available=True,
    pretrained_weights_available=True,
    colab_feasible_rating=4,
    colab_feasible_explanation='Should work with T4 GPU and gradient checkpointing',
    main_bottleneck='compute_time',
    improvements=json.dumps([
        {'improvement': 'Test on smaller datasets', 'rationale': 'Validate approach faster', 
         'demo_appeal': 'Quick results', 'effort': 'low', 'category': 'dataset'},
        {'improvement': 'Compare with standard transformer', 'rationale': 'Show improvements clearly',
         'demo_appeal': 'Clear benchmark', 'effort': 'medium', 'category': 'comparison'}
    ])
)

# For paper 2 (Efficient Training paper)
analyses_t.insert(
    fid=2, pid=2,
    gpu_vram_gb_min=80.0,
    training_time_hours=120.0,
    training_time_confidence='stated',
    multi_gpu_required=True,
    dataset_publicly_available=False,
    dataset_size_gb=250.0,
    code_available=False,
    pretrained_weights_available=False,
    colab_feasible_rating=1,
    colab_feasible_explanation='Requires multiple A100s and proprietary dataset',
    main_bottleneck='multi_gpu',
    improvements=json.dumps([
        {'improvement': 'Implement key algorithm on toy dataset', 'rationale': 'Demonstrate core technique',
         'demo_appeal': 'Shows understanding', 'effort': 'high', 'category': 'ablation'},
        {'improvement': 'Visualize training dynamics', 'rationale': 'Make results more interpretable',
         'demo_appeal': 'Eye-catching plots', 'effort': 'low', 'category': 'visualization'}
    ])
)



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


#--------------------------------#

def CardFooter(pid:int):
    return Div(
        A('Original', hx_get=f'/original?pid={pid}', hx_target='#abstract-text'),
        A('Simple', hx_get=f'/simple?pid={pid}', hx_target='#abstract-text', style='margin-left:10px;'),
        A('Compute', hx_get=f'/compute?pid={pid}', hx_target=f'#compute-{pid}', style='margin-left:10px;'),
        A('Improvements',hx_get=f'/improvements?pid={pid}', hx_target=f'#improvements-{pid}', style='margin-left:10px;')
    )

def PaperCard(meta:Metadata, pid:int):
    return Card(
        H4(A(meta.title, href=meta.url, style='color: #c66;')),
        P(meta.abstract, id='abstract-text'),
        Div(id=f'compute-{pid}'), Div(id=f'improvements-{pid}'), CardFooter(pid),
        style='background-color:#eee; padding:10px; border-radius:5px;')


@rt('/original')
def get(pid:int): return metadata_t[pid].abstract

@rt('/simple')
def get(pid:int): return metadata_t[pid].simple_abstract

@rt('/compute')
def get(pid:int):
    analysis = analyses_t[pid]
    return Div(format_compute(analysis),
               A('remove', hx_get=f'/remove/compute/{pid}', hx_swap='outerHTML', hx_target=f'#compute-{pid}', style='margin-left:10px;'),
               id=f'compute-{pid}')

@rt('/improvements')
def get(pid:int):
    analysis = analyses_t[pid]
    return Div(format_improvements(analysis), 
               A('remove', hx_get=f'/remove/improvements/{pid}', hx_swap='outerHTML', hx_target=f'#improvements-{pid}', style='margin-left:10px;'),
               id=f'improvements-{pid}'
              )

@rt('/remove/{section}/{pid}')
def remove(section:str, pid:int): return Div(id=f'{section}-{pid}')

limit = 1
more_link = A('Load More..', hx_get='/load_more', hx_swap='beforeend', hx_target='#papers')

@rt('/load_more')
def load_more(sess):
    offset = sess['offset']
    papers = metadata_t(limit=limit, offset=offset)
    cards = [PaperCard(p, i+offset+limit) for i, p in enumerate(papers)]
    sess['offset'] = offset + limit  # update for next batch
    return (*cards, )

@rt('/')
def index(sess):
    sess['offset'] = limit  # Set initial offset
    n = db.q('select count(*) from metadata')[0]['count(*)']
    papers = metadata_t(limit=limit, offset=0)
    cards = [PaperCard(p, i+1) for i, p in enumerate(papers)]
    return Titled('SimpleRead', Div(*cards, id='papers'), more_link)



serve()