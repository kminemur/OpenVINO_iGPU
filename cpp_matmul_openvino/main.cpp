#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset10.hpp>

namespace {
std::size_t parse_dim(const char* arg, const char* name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(arg, &end, 10);
    if (end == arg || *end != '\0' || value == 0ULL) {
        throw std::invalid_argument(std::string("Invalid ") + name + " dimension: " + arg);
    }
    return static_cast<std::size_t>(value);
}

void fill_input(std::vector<float>& data, float base) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = base + static_cast<float>(i % 17) * 0.1f;
    }
}

std::vector<float> cpu_matmul(const std::vector<float>& a,
                              const std::vector<float>& b,
                              std::size_t m,
                              std::size_t k,
                              std::size_t n) {
    std::vector<float> c(m * n, 0.0f);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const float a_val = a[i * k + p];
            for (std::size_t j = 0; j < n; ++j) {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    return c;
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string device = (argc > 1) ? argv[1] : "GPU";
        const std::size_t m = (argc > 2) ? parse_dim(argv[2], "M") : 2;
        const std::size_t k = (argc > 3) ? parse_dim(argv[3], "K") : 3;
        const std::size_t n = (argc > 4) ? parse_dim(argv[4], "N") : 4;
        const std::size_t cpu_loops = (argc > 5) ? parse_dim(argv[5], "CPU_LOOPS") : 1;

        auto a = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{m, k});
        a->set_friendly_name("A");
        auto b = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{k, n});
        b->set_friendly_name("B");

        auto matmul = std::make_shared<ov::opset10::MatMul>(a, b, false, false);
        matmul->set_friendly_name("Y");
        auto result = std::make_shared<ov::opset10::Result>(matmul);

        auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{a, b}, "MatMulModel");

        std::vector<float> a_data(m * k);
        std::vector<float> b_data(k * n);
        fill_input(a_data, 1.0f);
        fill_input(b_data, 0.5f);

        ov::Core core;
        ov::CompiledModel compiled = core.compile_model(model, device);
        ov::InferRequest req = compiled.create_infer_request();

        ov::Tensor a_tensor(ov::element::f32, ov::Shape{m, k}, a_data.data());
        ov::Tensor b_tensor(ov::element::f32, ov::Shape{k, n}, b_data.data());

        req.set_input_tensor(0, a_tensor);
        req.set_input_tensor(1, b_tensor);
        req.infer();

        const ov::Tensor y = req.get_output_tensor(0);
        const float* y_ptr = y.data<const float>();

        std::vector<float> reference;
        for (std::size_t i = 0; i < cpu_loops; ++i) {
            reference = cpu_matmul(a_data, b_data, m, k, n);
        }
        float max_abs_diff = 0.0f;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(reference[i] - y_ptr[i]));
        }

        std::cout << "Device: " << device << "\n";
        std::cout << "Shape: A[" << m << "x" << k << "] * B[" << k << "x" << n << "]\n";
        std::cout << "CPU reference loops: " << cpu_loops << "\n";
        std::cout << "Max abs diff vs CPU reference: " << max_abs_diff << "\n";

        const std::size_t print_rows = std::min<std::size_t>(m, 4);
        const std::size_t print_cols = std::min<std::size_t>(n, 8);
        std::cout << "Output (first " << print_rows << "x" << print_cols << " block):\n";
        for (std::size_t i = 0; i < print_rows; ++i) {
            for (std::size_t j = 0; j < print_cols; ++j) {
                std::cout << std::fixed << std::setprecision(3) << y_ptr[i * n + j] << " ";
            }
            std::cout << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        std::cerr << "Usage: matmul_openvino [DEVICE] [M] [K] [N] [CPU_LOOPS]\n";
        std::cerr << "Example: matmul_openvino GPU 256 512 128 10\n";
        return 1;
    }
}
