from fasthtml.common import *

app, rt = fast_app(live=True)


footer = Div(A('Original', hx_get='/original', hx_target='#abstract-text'),
             A('Simple', hx_get='/simple', hx_target='#abstract-text', style='margin-left:10px;'),
             A('Compute', hx_get='/compute', hx_target='#compute', style='margin-left:10px;'),
             A('Improvements',hx_get='/improvements', hx_target='#improvements', style='margin-left:10px;')
            )
c = Card(
    H4(A('Title', href='#', style='color: #c66;')),
    P('Some abstract text', id='abstract-text'),
    Div(id='compute'),
    Div(id='improvements'),
    footer=footer,
    style='background-color:#eee; padding:10px; border-radius:5px;')


@rt('/original')
def get(): return 'Original Academic Text...'

@rt('/simple')
def get(): return 'Simplified Text...'

@rt('/compute')
def get():
    return Div('Compute Requirements',
               A('remove', hx_get='/remove/compute', hx_swap='outerHTML', hx_target='#compute', style='margin-left:10px;'),
               id='compute')

@rt('/improvements')
def get():
    return Div('Suggested Improvements', 
               A('remove', hx_get='/remove/improvements', hx_swap='outerHTML', hx_target='#improvements', style='margin-left:10px;'),
               id='improvements'
              )

@rt('/remove/{section}')
def remove(section:str): return Div(id=section)

@rt('/')
def index():
    return Titled('SimpleRead', c)


serve()