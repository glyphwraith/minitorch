# minitorch

A tiny, header-only deep learning library in modern C++17 with a reverse-mode autograd engine.
Built from scratch to understand what PyTorch does under the hood.

## Features
- Dynamic computation graph with reverse-mode automatic differentiation
- Ops: `matmul`, `add`, `add_bias` (broadcast), `relu`, `tanh`, `sigmoid`, `mse`, `softmax_cross_entropy`
- Multithreaded matrix multiplication (forward and backward)
- `Linear` layer, `MLP` model, He initialization
- Adam optimizer
- Zero dependencies (just the standard library + pthreads)

## Build and run
```bash
cmake -B build
cmake --build build
./build/spiral   # trains an MLP on a 3-arm spiral, prints an ASCII decision map
```

## Usage
```cpp
#include "minitorch.hpp"
using namespace mt;

MLP net({2, 64, 64, 3}, Act::ReLU);
Adam opt(net.parameters(), 0.01f);

for (int epoch = 0; epoch < 400; ++epoch) {
    opt.zero_grad();
    Var loss = softmax_cross_entropy(net(X), labels);
    backward(loss);
    opt.step();
}
```

## How autograd works
Every op creates a `Node` holding its values, its gradient buffer, its parent nodes, and a
`back` closure. `backward(loss)` topologically sorts the graph, seeds the loss gradient with 1,
and runs each node's closure in reverse order, accumulating gradients into its parents.

## Ideas for extending
- Add `conv2d`, `batchnorm`, `dropout`
- SIMD (AVX2) or cache-blocked GEMM
- MNIST loader and a training example
- Serialize and load model weights
- SGD with momentum, learning-rate schedulers

## License
MIT
