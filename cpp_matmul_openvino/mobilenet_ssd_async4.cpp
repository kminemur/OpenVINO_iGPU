#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <openvino/openvino.hpp>

namespace {
std::atomic<bool> g_run{true};

void signal_handler(int) {
    g_run.store(false);
}

void fill_random(ov::Tensor& tensor, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float* data = tensor.data<float>();
    const std::size_t n = tensor.get_size();
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: mobilenet_ssd_async4 <model.xml> [DEVICE]\n";
            return 1;
        }

        const std::string model_path = argv[1];
        const std::string device = (argc > 2) ? argv[2] : "GPU";
        constexpr std::size_t kNumRequests = 4;

        std::signal(SIGINT, signal_handler);

        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        ov::CompiledModel compiled = core.compile_model(model, device);

        const ov::Output<const ov::Node> input_port = compiled.input();
        const ov::Shape input_shape = input_port.get_shape();
        const ov::element::Type input_type = input_port.get_element_type();
        if (input_type != ov::element::f32) {
            throw std::runtime_error("This sample expects FP32 input model.");
        }

        std::vector<ov::InferRequest> requests;
        requests.reserve(kNumRequests);
        std::vector<ov::Tensor> input_tensors;
        input_tensors.reserve(kNumRequests);
        for (std::size_t i = 0; i < kNumRequests; ++i) {
            requests.emplace_back(compiled.create_infer_request());
            input_tensors.emplace_back(ov::Tensor(input_type, input_shape));
            requests.back().set_input_tensor(input_tensors.back());
        }

        std::atomic<unsigned long long> completed{0};
        std::atomic<unsigned int> seed_counter{100};

        for (std::size_t i = 0; i < kNumRequests; ++i) {
            requests[i].set_callback([&, i](std::exception_ptr ex) {
                if (ex) {
                    try {
                        std::rethrow_exception(ex);
                    } catch (const std::exception& e) {
                        std::cerr << "Request " << i << " failed: " << e.what() << "\n";
                    }
                    g_run.store(false);
                    return;
                }

                ++completed;
                if (!g_run.load()) {
                    return;
                }

                std::mt19937 rng(seed_counter.fetch_add(1));
                fill_random(input_tensors[i], rng);
                requests[i].start_async();
            });
        }

        for (std::size_t i = 0; i < kNumRequests; ++i) {
            std::mt19937 rng(seed_counter.fetch_add(1));
            fill_random(input_tensors[i], rng);
            requests[i].start_async();
        }

        std::cout << "Started async inference with " << kNumRequests << " concurrent requests on " << device << "\n";
        std::cout << "Model: " << model_path << "\n";
        std::cout << "Press Ctrl+C to stop.\n";

        unsigned long long last = 0;
        while (g_run.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const unsigned long long now = completed.load();
            std::cout << "completed_total=" << now << ", infer/s=" << (now - last) << "\n";
            last = now;
        }

        for (auto& req : requests) {
            req.wait();
        }

        std::cout << "Stopped. Total completed: " << completed.load() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
