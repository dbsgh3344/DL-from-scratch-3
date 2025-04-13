import os
import subprocess
from graphviz import Source

def _dot_var(v, verbose=False):    
    dot_var = '{} [label="{}", color=black, style=filled, fillcolor=lightblue]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += f"{v.shape} {v.dtype}"
    return dot_var.format(id(v), name)

def _dot_func(f, verbose=False):
    dot_func = '{} [label="{}", color=lightblue, style=filled, fillcolor=lightgreen, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=False):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output,verbose=verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func,verbose=verbose)
        for x in func.inputs:
            txt += _dot_var(x,verbose=verbose)
            
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'


def plot_dot_graph_using_lib(output, verbose=False, to_file='graph.png'):    
    dot_graph = get_dot_graph(output, verbose=verbose)
    extension = os.path.splitext(to_file)[1][1:]
    src = Source(dot_graph)
    src.format = extension    
    src.render(to_file, cleanup=True)

    

def plot_dot_graph(output, verbose=False, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose=verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'),'.dezero')
    os.makedirs(tmp_dir, exist_ok=True)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    
    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd,shell=True)


    

if __name__ == '__main__':
    
    # plot_dot_graph(None, verbose=True)
    
    print(os.path.expanduser('~'))
    a= "test/graph.png"
    # print(os.path.splitext(a))
    