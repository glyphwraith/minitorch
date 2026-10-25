// Demo: train an MLP on the classic 3-arm spiral dataset.
#include <chrono>
#include <iomanip>
#include "minitorch.hpp"

using namespace mt;

struct Dataset { Var X; std::vector<int> y; };

Dataset make_spiral(int per_class, int classes, float noise = 0.2f) {
    Dataset d{make(per_class * classes, 2), std::vector<int>(per_class * classes)};
    std::normal_distribution<float> nd(0.f, noise);
    for (int k = 0; k < classes; ++k)
        for (int i = 0; i < per_class; ++i) {
            float r = (float)i / per_class;
            float t = k * 4.f + 4.f * r + nd(rng());
            int idx = k * per_class + i;
            d.X->v[idx * 2] = r * std::sin(t);
            d.X->v[idx * 2 + 1] = r * std::cos(t);
            d.y[idx] = k;
        }
    return d;
}

void ascii_boundary(const MLP& net, const Dataset& d) {
    const int W = 60, H = 28;
    const char* shades = ".o#";
    std::cout << "\nDecision map (background = predicted class, X = a training point):\n";
    std::vector<std::string> grid(H, std::string(W, ' '));
    Var pts = make(W * H, 2);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            pts->v[(y * W + x) * 2] = -1.1f + 2.2f * x / (W - 1);
            pts->v[(y * W + x) * 2 + 1] = 1.1f - 2.2f * y / (H - 1);
        }
    Var out = net(pts);
    for (int i = 0; i < W * H; ++i) {
        const float* row = &out->v[i * 3];
        grid[i / W][i % W] = shades[std::max_element(row, row + 3) - row];
    }
    for (int i = 0; i < d.X->r; ++i) {
        int x = (int)((d.X->v[i * 2] + 1.1f) / 2.2f * (W - 1) + 0.5f);
        int y = (int)((1.1f - d.X->v[i * 2 + 1]) / 2.2f * (H - 1) + 0.5f);
        if (x >= 0 && x < W && y >= 0 && y < H) grid[y][x] = 'X';
    }
    for (auto& row : grid) std::cout << row << "\n";
}

int main() {
    Dataset data = make_spiral(100, 3);
    MLP net({2, 64, 64, 3}, Act::ReLU);
    Adam opt(net.parameters(), 0.01f);

    std::cout << "minitorch: training 2-64-64-3 MLP on spiral (" << data.X->r << " points)\n";
    auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 1; epoch <= 400; ++epoch) {
        opt.zero_grad();
        Var logits = net(data.X);
        Var loss = softmax_cross_entropy(logits, data.y);
        backward(loss);
        opt.step();
        if (epoch % 50 == 0 || epoch == 1)
            std::cout << "epoch " << std::setw(3) << epoch << "  loss " << std::fixed << std::setprecision(4)
                      << loss->v[0] << "  acc " << std::setprecision(1) << 100.f * accuracy(logits, data.y) << "%\n";
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "done in " << std::setprecision(2) << secs << "s\n";
    ascii_boundary(net, data);
    return 0;
}
