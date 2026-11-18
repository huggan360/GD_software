#include "classifier.hpp"
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <opencv2/imgproc.hpp>

struct Model::State {
    torch::jit::Module module;
};

Model::Model(const fs::path& path, const std::string& format) : state(std::make_unique<State>()) {
    if (!fs::is_regular_file(path)) throw std::runtime_error("Missing compiled model: " + path.string());
    at::set_num_threads(4);
    torch::jit::ExtraFilesMap extra{{"metadata.json", ""}};
    state->module = torch::jit::load(path.string(), torch::kCPU, extra);
    state->module.eval();
    auto metadata = Json::parse(extra.at("metadata.json"));
    if (metadata.at("format") != format) throw std::runtime_error("Wrong model format: export it from the matching GD training repo.");
    classes = metadata.at("classes").get<std::vector<std::string>>();
    if (classes.empty()) throw std::runtime_error("The model has no class labels.");
    image_size = metadata.value("image_size", 640);
}

Model::~Model() = default;

std::vector<float> Model::classify(const cv::Mat& rgb) {
    c10::InferenceMode guard;
    auto input = torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kUInt8);
    auto probabilities = state->module.forward({input}).toTensor().softmax(1).contiguous();
    if (probabilities.numel() != static_cast<int64_t>(classes.size())) throw std::runtime_error("ViT output does not match its class labels.");
    return {probabilities.data_ptr<float>(), probabilities.data_ptr<float>() + classes.size()};
}

std::vector<Detection> Model::detect(const cv::Mat& rgb, float confidence) {
    c10::InferenceMode guard;
    float scale;
    int left, top;
    cv::Mat image = letterbox(rgb, image_size, image_size, scale, left, top);
    auto input = torch::from_blob(image.data, {1, image_size, image_size, 3}, torch::kUInt8).permute({0, 3, 1, 2}).to(torch::kFloat32) / 255.0;
    auto output = state->module.forward({input});
    if (output.isTuple()) output = output.toTuple()->elements()[0];
    auto rows = output.toTensor().to(torch::kCPU).contiguous();
    if (rows.dim() != 3 || rows.size(0) != 1) throw std::runtime_error("Expected a single-image YOLO output.");
    bool raw = rows.size(1) == static_cast<int64_t>(4 + classes.size());
    if (raw) rows = rows.transpose(1, 2).contiguous();
    else if (rows.size(2) != 6) throw std::runtime_error("Expected YOLO boxes and class scores, or final detections.");
    auto values = rows.accessor<float, 3>();
    std::vector<Detection> result;
    for (int64_t index = 0; index < rows.size(1); ++index) {
        const auto row = values[0][index];
        int label = raw ? 0 : int(row[5]);
        float score = row[4];
        if (raw) for (size_t index = 1; index < classes.size(); ++index) if (row[4 + index] > score) { score = row[4 + index]; label = int(index); }
        if (score < confidence) continue;
        float x1 = std::clamp(((raw ? row[0] - row[2] / 2 : row[0]) - left) / scale, 0.0f, float(rgb.cols));
        float y1 = std::clamp(((raw ? row[1] - row[3] / 2 : row[1]) - top) / scale, 0.0f, float(rgb.rows));
        float x2 = std::clamp(((raw ? row[0] + row[2] / 2 : row[2]) - left) / scale, 0.0f, float(rgb.cols));
        float y2 = std::clamp(((raw ? row[1] + row[3] / 2 : row[3]) - top) / scale, 0.0f, float(rgb.rows));
        if (x2 > x1 && y2 > y1) result.push_back({{x1, y1, x2 - x1, y2 - y1}, score, label});
    }
    return result;
}
