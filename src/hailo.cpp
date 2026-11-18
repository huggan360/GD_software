#include "classifier.hpp"
#include <hailo/hailort.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <thread>

using namespace hailort;

// Keep Hailo errors visible instead of falling back to a slow CPU model.
template<class T> T checked(Expected<T>&& value) {
    if (!value) throw std::runtime_error("HailoRT error " + std::to_string(value.status()));
    return value.release();
}

class HailoDetector : public Detector {
public:
    explicit HailoDetector(const Settings& settings) : hef(checked(Hef::create(settings.yolo.string()))) {
        std::ifstream labels(settings.labels);
        std::string name;
        while (std::getline(labels, name)) { if (!name.empty() && name.back() == '\r') name.pop_back(); if (!name.empty()) classes.push_back(name); }
        if (classes.empty() || std::set<std::string>(classes.begin(), classes.end()).size() != classes.size()) throw std::runtime_error("yolo/classes.txt must contain unique labels in model order.");
        confidence = settings.config["detector"]["confidence"].as<float>();
        hailo_vdevice_params_t params;
        auto status = hailo_init_vdevice_params(&params);
        if (status != HAILO_SUCCESS) throw std::runtime_error("Cannot initialize Hailo device settings.");
        params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;
        // The annotation tool and orchestrator take turns using the accelerator.
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
        while (!device) {
            auto available = VDevice::create(params);
            if (available) device = available.release();
            else if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Cannot reserve Hailo-8: " + std::to_string(available.status()));
            else std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        auto groups = checked(device->configure(hef));
        if (groups.size() != 1) throw std::runtime_error("Expected one YOLO network group.");
        network = groups[0];
        auto inputs = checked(hef.get_input_vstream_infos());
        outputs = checked(hef.get_output_vstream_infos());
        if (inputs.size() != 1 || inputs[0].shape.features != 3 || outputs.size() != 6) throw std::runtime_error("Use the YOLO26 six-output Hailo-8 export from GD_tracking.");
        input_name = inputs[0].name;
        width = inputs[0].shape.width;
        height = inputs[0].shape.height;
        auto in_params = checked(network->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE));
        auto out_params = checked(network->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE));
        for (auto& entry : in_params) entry.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        for (auto& entry : out_params) entry.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        pipeline = std::make_unique<InferVStreams>(checked(InferVStreams::create(*network, in_params, out_params)));
        for (const auto& info : outputs) {
            size_t bytes = checked(pipeline->get_output_by_name(info.name)).get().get_frame_size();
            buffers[info.name].resize(bytes / sizeof(float));
        }
        active = checked(network->activate());
    }

    std::vector<Detection> detect(const cv::Mat& rgb) override {
        float scale;
        int left, top;
        cv::Mat image = letterbox(rgb, width, height, scale, left, top);
        std::map<std::string, MemoryView> in{{input_name, MemoryView(image.data, image.total() * image.elemSize())}}, out;
        for (auto& entry : buffers) out.emplace(entry.first, MemoryView(entry.second.data(), entry.second.size() * sizeof(float)));
        auto status = pipeline->infer(in, out, 1);
        if (status != HAILO_SUCCESS) throw std::runtime_error("Hailo inference failed: " + std::to_string(status));

        // Each scale has four box distances and one score per detector class.
        std::vector<Detection> rows;
        for (const auto& boxes : outputs) {
            if (boxes.shape.features != 4) continue;
            const hailo_vstream_info_t* scores = nullptr;
            for (const auto& item : outputs) if (std::string(item.name) != boxes.name && item.shape.height == boxes.shape.height && item.shape.width == boxes.shape.width && item.shape.features == classes.size()) scores = &item;
            if (!scores) throw std::runtime_error("The HEF box and class outputs do not match classes.txt.");
            const auto& box_data = buffers.at(boxes.name);
            const auto& class_data = buffers.at(scores->name);
            bool logits = std::any_of(class_data.begin(), class_data.end(), [](float v) { return v < 0 || v > 1; });
            for (unsigned y = 0; y < boxes.shape.height; ++y) for (unsigned x = 0; x < boxes.shape.width; ++x) {
                size_t cell = y * boxes.shape.width + x;
                auto first = class_data.begin() + cell * classes.size();
                auto best = std::max_element(first, first + classes.size());
                float score = logits ? 1.0f / (1.0f + std::exp(-*best)) : *best;
                if (score < confidence) continue;
                const float* box = box_data.data() + cell * 4;
                float stride_x = float(width) / boxes.shape.width, stride_y = float(height) / boxes.shape.height;
                float x1 = std::clamp(((x + 0.5f - box[0]) * stride_x - left) / scale, 0.0f, float(rgb.cols));
                float y1 = std::clamp(((y + 0.5f - box[1]) * stride_y - top) / scale, 0.0f, float(rgb.rows));
                float x2 = std::clamp(((x + 0.5f + box[2]) * stride_x - left) / scale, 0.0f, float(rgb.cols));
                float y2 = std::clamp(((y + 0.5f + box[3]) * stride_y - top) / scale, 0.0f, float(rgb.rows));
                if (x2 > x1 && y2 > y1) rows.push_back({{x1, y1, x2 - x1, y2 - y1}, score, int(best - first)});
            }
        }
        return rows;
    }
private:
    Hef hef;
    std::unique_ptr<VDevice> device;
    std::shared_ptr<ConfiguredNetworkGroup> network;
    std::unique_ptr<ActivatedNetworkGroup> active;
    std::unique_ptr<InferVStreams> pipeline;
    std::vector<hailo_vstream_info_t> outputs;
    std::map<std::string, std::vector<float>> buffers;
    std::string input_name;
    int width = 640, height = 640;
    float confidence = 0.15;
};

std::unique_ptr<Detector> make_hailo_detector(const Settings& settings) { return std::make_unique<HailoDetector>(settings); }
