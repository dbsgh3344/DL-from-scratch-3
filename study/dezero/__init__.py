import sys, os
sys.path.append(os.path.join(os.getcwd(),"study"))
is_simple_core = False

from dezero.utils import _dot_var, _dot_func, get_dot_graph, plot_dot_graph, plot_dot_graph_using_lib
from dezero.layers import Layer
from dezero.models import Model
from dezero import optimizers

if is_simple_core:
    from dezero.core_simple import Variable, Function, use_config, no_grad, as_array, as_variable,setup_variable
else :    
    from dezero.core import Variable,Function, use_config, no_grad, as_array, as_variable , setup_variable , Parameter    

setup_variable()