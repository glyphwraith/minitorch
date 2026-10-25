// minitorch.hpp - a tiny header-only deep learning library with reverse-mode autograd.
// C++17, no dependencies.
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace mt {

// ---------------------------------------------------------------- graph node
struct Node;
using Var = std::shared_ptr<Node>;

struct Node {
    int r = 0, c = 0;                    // shape (rows x cols), row-major
    std::vector<float> v, g;             // values and gradients
    std::vector<Var> parents;            // inputs that produced this node
    std::function<void(Node&)> back;     // propagates self.g into parents' g
    bool requires_grad = false;

    Node(int rows, int cols) : r(rows), c(cols), v((size_t)rows * cols, 0.f), g((size_t)rows * cols, 0.f) {}
    size_t size() const { return v.size(); }
};

inline Var make(int r, int c, bool requires_grad = false) {
    auto n = std::make_shared<Node>(r, c);
    n->requires_grad = requires_grad;
    return n;
}

inline std::mt19937& rng() {
    static std::mt19937 gen(1337);
    return gen;
}

// ---------------------------------------------------------------- threading
inline void parallel_for(int n, const std::function<void(int, int)>& fn) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned t = std::min<unsigned>(hw, (unsigned)std::max(1, n / 8));
    if (t <= 1) { fn(0, n); return; }
    std::vector<std::thread> pool;
    int chunk = (n + (int)t - 1) / (int)t;
    for (unsigned i = 0; i < t; ++i) {
        int s = (int)i * chunk, e = std::min(n, s + chunk);
        if (s >= e) break;
        pool.emplace_back(fn, s, e);
    }
    for (auto& th : pool) th.join();
}

// C (m x n) += op(A) (m x k) * op(B) (k x n); tA/tB say whether the stored matrix is transposed.
inline void gemm(const float* A, const float* B, float* C, int m, int k, int n, bool tA, bool tB) {
    auto work = [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i)
            for (int p = 0; p < k; ++p) {
                float a = tA ? A[(size_t)p * m + i] : A[(size_t)i * k + p];
                if (a == 0.f) continue;
                for (int j = 0; j < n; ++j) {
                    float b = tB ? B[(size_t)j * k + p] : B[(size_t)p * n + j];
                    C[(size_t)i * n + j] += a * b;
                }
            }
    };
    if ((long long)m * k * n > 20000) parallel_for(m, work);
    else work(0, m);
}

// ---------------------------------------------------------------- operations
inline Var matmul(const Var& a, const Var& b) {
    if (a->c != b->r) throw std::runtime_error("matmul: shape mismatch");
    Var out = make(a->r, b->c, a->requires_grad || b->requires_grad);
    out->parents = {a, b};
    gemm(a->v.data(), b->v.data(), out->v.data(), a->r, a->c, b->c, false, false);
    out->back = [](Node& s) {
        Node &A = *s.parents[0], &B = *s.parents[1];
        if (A.requires_grad) gemm(s.g.data(), B.v.data(), A.g.data(), A.r, s.c, A.c, false, true);
        if (B.requires_grad) gemm(A.v.data(), s.g.data(), B.g.data(), B.r, A.r, B.c, true, false);
    };
    return out;
}

// x (n x c) + bias (1 x c), broadcast over rows
inline Var add_bias(const Var& x, const Var& bias) {
    if (bias->r != 1 || bias->c != x->c) throw std::runtime_error("add_bias: shape mismatch");
    Var out = make(x->r, x->c, x->requires_grad || bias->requires_grad);
    out->parents = {x, bias};
    for (int i = 0; i < x->r; ++i)
        for (int j = 0; j < x->c; ++j) out->v[i * x->c + j] = x->v[i * x->c + j] + bias->v[j];
    out->back = [](Node& s) {
        Node &X = *s.parents[0], &B = *s.parents[1];
        for (int i = 0; i < s.r; ++i)
            for (int j = 0; j < s.c; ++j) {
                float g = s.g[i * s.c + j];
                if (X.requires_grad) X.g[i * s.c + j] += g;
                if (B.requires_grad) B.g[j] += g;
            }
    };
    return out;
}

inline Var add(const Var& a, const Var& b) {
    if (a->r != b->r || a->c != b->c) throw std::runtime_error("add: shape mismatch");
    Var out = make(a->r, a->c, a->requires_grad || b->requires_grad);
    out->parents = {a, b};
    for (size_t i = 0; i < out->size(); ++i) out->v[i] = a->v[i] + b->v[i];
    out->back = [](Node& s) {
        for (auto& p : s.parents)
            if (p->requires_grad)
                for (size_t i = 0; i < s.size(); ++i) p->g[i] += s.g[i];
    };
    return out;
}

// element-wise unary op helper: f(x) and df(x, y)
inline Var unary(const Var& x, float (*f)(float), float (*df)(float, float)) {
    Var out = make(x->r, x->c, x->requires_grad);
    out->parents = {x};
    for (size_t i = 0; i < out->size(); ++i) out->v[i] = f(x->v[i]);
    out->back = [df](Node& s) {
        Node& X = *s.parents[0];
        if (!X.requires_grad) return;
        for (size_t i = 0; i < s.size(); ++i) X.g[i] += s.g[i] * df(X.v[i], s.v[i]);
    };
    return out;
}

inline Var relu(const Var& x) {
    return unary(x, [](float a) { return a > 0.f ? a : 0.f; }, [](float a, float) { return a > 0.f ? 1.f : 0.f; });
}
inline Var tanh_(const Var& x) {
    return unary(x, [](float a) { return std::tanh(a); }, [](float, float y) { return 1.f - y * y; });
}
inline Var sigmoid(const Var& x) {
    return unary(x, [](float a) { return 1.f / (1.f + std::exp(-a)); }, [](float, float y) { return y * (1.f - y); });
}

// mean squared error -> 1x1 scalar
inline Var mse(const Var& pred, const Var& target) {
    if (pred->r != target->r || pred->c != target->c) throw std::runtime_error("mse: shape mismatch");
    Var out = make(1, 1, true);
    out->parents = {pred, target};
    double acc = 0;
    for (size_t i = 0; i < pred->size(); ++i) { double d = pred->v[i] - target->v[i]; acc += d * d; }
    out->v[0] = (float)(acc / pred->size());
    out->back = [](Node& s) {
        Node &P = *s.parents[0], &T = *s.parents[1];
        float k = 2.f * s.g[0] / (float)P.size();
        for (size_t i = 0; i < P.size(); ++i) {
            float d = k * (P.v[i] - T.v[i]);
            if (P.requires_grad) P.g[i] += d;
            if (T.requires_grad) T.g[i] -= d;
        }
    };
    return out;
}

// softmax + cross-entropy over rows of logits, integer class labels -> 1x1 scalar
inline Var softmax_cross_entropy(const Var& logits, const std::vector<int>& labels) {
    int n = logits->r, c = logits->c;
    if ((int)labels.size() != n) throw std::runtime_error("cross_entropy: label count mismatch");
    Var out = make(1, 1, true);
    out->parents = {logits};
    auto probs = std::make_shared<std::vector<float>>((size_t)n * c);
    double loss = 0;
    for (int i = 0; i < n; ++i) {
        const float* row = &logits->v[(size_t)i * c];
        float mx = *std::max_element(row, row + c);
        double z = 0;
        for (int j = 0; j < c; ++j) z += std::exp(row[j] - mx);
        for (int j = 0; j < c; ++j) (*probs)[(size_t)i * c + j] = (float)(std::exp(row[j] - mx) / z);
        loss -= std::log(std::max(1e-12f, (*probs)[(size_t)i * c + labels[i]]));
    }
    out->v[0] = (float)(loss / n);
    out->back = [probs, labels, n, c](Node& s) {
        Node& L = *s.parents[0];
        if (!L.requires_grad) return;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < c; ++j) {
                float y = (j == labels[i]) ? 1.f : 0.f;
                L.g[(size_t)i * c + j] += s.g[0] * ((*probs)[(size_t)i * c + j] - y) / (float)n;
            }
    };
    return out;
}

// ---------------------------------------------------------------- autograd driver
inline void backward(const Var& root) {
    std::vector<Node*> order;
    std::unordered_set<Node*> seen;
    std::function<void(Node*)> topo = [&](Node* n) {
        if (!seen.insert(n).second) return;
        for (auto& p : n->parents) topo(p.get());
        order.push_back(n);
    };
    topo(root.get());
    std::fill(root->g.begin(), root->g.end(), 1.f);
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if ((*it)->back) (*it)->back(**it);
}

// ---------------------------------------------------------------- neural network pieces
struct Linear {
    Var W, b;
    Linear(int in, int out) : W(make(in, out, true)), b(make(1, out, true)) {
        std::normal_distribution<float> nd(0.f, std::sqrt(2.f / in));  // He init
        for (auto& x : W->v) x = nd(rng());
    }
    Var operator()(const Var& x) const { return add_bias(matmul(x, W), b); }
};

enum class Act { ReLU, Tanh, Sigmoid };

struct MLP {
    std::vector<Linear> layers;
    Act act;
    MLP(const std::vector<int>& sizes, Act a = Act::ReLU) : act(a) {
        for (size_t i = 0; i + 1 < sizes.size(); ++i) layers.emplace_back(sizes[i], sizes[i + 1]);
    }
    Var operator()(Var x) const {
        for (size_t i = 0; i < layers.size(); ++i) {
            x = layers[i](x);
            if (i + 1 < layers.size()) x = act == Act::ReLU ? relu(x) : act == Act::Tanh ? tanh_(x) : sigmoid(x);
        }
        return x;
    }
    std::vector<Var> parameters() const {
        std::vector<Var> ps;
        for (auto& l : layers) { ps.push_back(l.W); ps.push_back(l.b); }
        return ps;
    }
};

// ---------------------------------------------------------------- optimizers
struct Adam {
    std::vector<Var> params;
    float lr, b1, b2, eps;
    int t = 0;
    std::vector<std::vector<float>> m, s;
    Adam(std::vector<Var> p, float lr_ = 1e-2f, float b1_ = 0.9f, float b2_ = 0.999f, float eps_ = 1e-8f)
        : params(std::move(p)), lr(lr_), b1(b1_), b2(b2_), eps(eps_) {
        for (auto& p2 : params) { m.emplace_back(p2->size(), 0.f); s.emplace_back(p2->size(), 0.f); }
    }
    void zero_grad() { for (auto& p : params) std::fill(p->g.begin(), p->g.end(), 0.f); }
    void step() {
        ++t;
        float c1 = 1.f - std::pow(b1, (float)t), c2 = 1.f - std::pow(b2, (float)t);
        for (size_t k = 0; k < params.size(); ++k) {
            Node& p = *params[k];
            for (size_t i = 0; i < p.size(); ++i) {
                m[k][i] = b1 * m[k][i] + (1.f - b1) * p.g[i];
                s[k][i] = b2 * s[k][i] + (1.f - b2) * p.g[i] * p.g[i];
                p.v[i] -= lr * (m[k][i] / c1) / (std::sqrt(s[k][i] / c2) + eps);
            }
        }
    }
};

// ---------------------------------------------------------------- utilities
inline float accuracy(const Var& logits, const std::vector<int>& labels) {
    int ok = 0;
    for (int i = 0; i < logits->r; ++i) {
        const float* row = &logits->v[(size_t)i * logits->c];
        int best = (int)(std::max_element(row, row + logits->c) - row);
        ok += best == labels[i];
    }
    return (float)ok / (float)logits->r;
}

}  // namespace mt
