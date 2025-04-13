import sys, os
sys.path.append(os.path.join(os.getcwd(),"study"))
is_simple_core = True

if is_simple_core:
    from dezero.core_simple import Variable, Function, use_config, no_grad, as_array, as_variable,setup_variable
    from dezero.utils import _dot_var, _dot_func, get_dot_graph, plot_dot_graph, plot_dot_graph_using_lib
else :
    pass
    # from dezero.core import Variable, Function, use_config, no_grad, as_array, as_variable
    # from dezero.utils import  Config, setup_variable

setup_variable()