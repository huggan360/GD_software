#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;
using Json = nlohmann::json;

struct Detection {
    cv::Rect2f box;
    float score;
    int label;
};

struct Settings {
    YAML::Node config;
    fs::path model_root, vit, yolo, labels, zones_file, acquisition;
    std::string backend;
};

struct Options {
    fs::path request, output, config, models;
    std::string record_id, mode;
    bool managed_retention = false;
};

// The model archive carries its labels and input format alongside the weights.
class Model {
public:
    Model(const fs::path& path, const std::string& format);
    ~Model();
    std::vector<float> classify(const cv::Mat& rgb);
    std::vector<Detection> detect(const cv::Mat& rgb, float confidence);
    std::vector<std::string> classes;
    int image_size = 640;
private:
    struct State;
    std::unique_ptr<State> state;
};

class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& rgb) = 0;
    std::vector<std::string> classes;
};

std::unique_ptr<Detector> make_detector(const Settings& settings);
#ifdef GD_HAILO
std::unique_ptr<Detector> make_hailo_detector(const Settings& settings);
#endif
cv::Mat letterbox(const cv::Mat& rgb, int width, int height, float& scale, int& left, int& top);
cv::Mat prepare_detector(const cv::Mat& rgb, bool enabled);
std::vector<Detection> suppress_duplicates(std::vector<Detection> rows, int limit);
Json read_json(const fs::path& path);
void write_json(const fs::path& path, const Json& value);
std::string now();
Settings load_settings(const Options& options);
Json run_clip(const Options& options, const Settings& settings);
