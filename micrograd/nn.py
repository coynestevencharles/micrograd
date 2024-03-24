import random
from micrograd.engine import Value

ACTIVATION_FN = "ReLU"

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, num_inputs, nonlin=True):
        self.weights = [Value(random.uniform(-1, 1)) for i in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
        self.activation_fn = ACTIVATION_FN if nonlin else None

    def __call__(self, x):
        # pre-activation value z = w * x + b
        z = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        # non-linear activation function
        if self.nonlin:
            if self.activation_fn == "ReLU":
                out = z.relu()
            elif self.activation_fn == "tanh":
                out = z.tanh()
            else:
                raise ValueError(f"Unknown activation function {self.activation_fn}")
        else:
            out = z
        return out
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f"{self.activation_fn}Neuron({len(self.weights)})"
    
class Layer(Module):

    def __init__(self, num_inputs, num_neurons, **kwargs):
        self.neurons = [Neuron(num_inputs, **kwargs) for i in range(num_neurons)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Module):

    def __init__(self, num_inputs, layer_sizes):
        sz = [num_inputs] + layer_sizes # [num_inputs, layer_sizes[0], layer_sizes[1], ...]
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(layer_sizes)-1) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"