#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

cv::Mat letterbox(const cv::Mat& rgb, int width, int height, float& scale, int& left, int& top) {
    scale = std::min(float(width) / rgb.cols, float(height) / rgb.rows);
    cv::Mat resized, image(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(rgb, resized, {int(std::nearbyint(rgb.cols * scale)), int(std::nearbyint(rgb.rows * scale))});
    left = (width - resized.cols) / 2;
    top = (height - resized.rows) / 2;
    resized.copyTo(image(cv::Rect(left, top, resized.cols, resized.rows)));
    return image;
}

cv::Mat prepare_detector(const cv::Mat& rgb, bool enabled) {
    if (!enabled) return rgb;
    cv::Mat lab, blurred;
    cv::cvtColor(rgb, lab, cv::COLOR_RGB2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    cv::createCLAHE(3.0, {8, 8})->apply(channels[0], channels[0]);
    cv::GaussianBlur(channels[0], blurred, {}, 1.0);
    cv::addWeighted(channels[0], 1.35, blurred, -0.35, 0, channels[0]);
    cv::merge(channels, lab);
    cv::cvtColor(lab, lab, cv::COLOR_Lab2RGB);
    return lab;
}

std::vector<Detection> suppress_duplicates(std::vector<Detection> rows, int limit) {
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const auto& row : rows) {
        bool duplicate = false;
        for (const auto& other : kept) {
            float intersection = (row.box & other.box).area();
            float iou = intersection / std::max(1e-6f, row.box.area() + other.box.area() - intersection);
            if (row.label == other.label && iou > 0.7f) { duplicate = true; break; }
        }
        if (!duplicate) kept.push_back(row);
        if (static_cast<int>(kept.size()) >= limit) break;
    }
    return kept;
}

class CpuDetector : public Detector {
public:
    explicit CpuDetector(const Settings& settings) : model(settings.yolo, "gd-yolo26-v1"), confidence(settings.config["detector"]["confidence"].as<float>()) { classes = model.classes; }
    std::vector<Detection> detect(const cv::Mat& rgb) override { return model.detect(rgb, confidence); }
private:
    Model model;
    float confidence;
};

std::unique_ptr<Detector> make_detector(const Settings& settings) {
    if (settings.backend == "torch") return std::make_unique<CpuDetector>(settings);
#ifdef GD_HAILO
    if (settings.backend == "hailo") return make_hailo_detector(settings);
#endif
    throw std::runtime_error("Hailo inference is unavailable in this build. Build with GD_HAILO=ON for the Pi.");
}
