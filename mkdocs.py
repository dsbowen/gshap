from docstr_md.python import PySoup, compile_md
from docstr_md.src_href import Github

src_href = Github('https://github.com/dsbowen/gshap/blob/master')

soup = PySoup(path='gshap/__init__.py', parser='sklearn', src_href=src_href)
soup.rm_properties()
compile_md(soup, compiler='sklearn', outfile='docs_md/kernel_explainer.md')

g_functions = ('hypothesis', 'intergroup', 'probability_distance')
for g in g_functions:
    soup = PySoup(
        path='gshap/{}.py'.format(g), 
        parser='sklearn', 
        src_href=src_href
    )
    compile_md(soup, compiler='sklearn', outfile='docs_md/{}.md'.format(g))
    
soup = PySoup(
    path='gshap/datasets/__init__.py', 
    parser='sklearn', 
    src_href=src_href
)
compile_md(soup, compiler='sklearn', outfile='docs_md/datasets.md')