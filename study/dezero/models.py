import dezero.layers as L
from dezero import utils
# from dezero.layers import Layer
from dezero import functions as F



class Model(L.Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph_using_lib(y, verbose=True, to_file=to_file)



class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid_simple):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, output_size in enumerate(fc_output_sizes):
            layer = L.Linear(output_size)
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x):
        
        for l in self.layers[:-1]:
            x = l(x)
            x = self.activation(x)
        return self.layers[-1](x)