#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>


//-------------------
// PRE PROCESSING
//-------------------

// Gör bilden skala upp
cv::Mat resize_for_yolo(const cv::Mat& image, int width, int height, float& scale, int& left, int& top) {
    scale = std::min(float(width) / image.cols, float(height) / image.rows);
    cv::Mat resized;
    cv::resize(image, resized, {int(std::nearbyint(image.cols * scale)), int(std::nearbyint(image.rows * scale))});
    left = (width - resized.cols) / 2;
    top = (height - resized.rows) / 2;
    cv::Mat padded(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(left, top, resized.cols, resized.rows)));
    return padded; //returnerar en bild
}

// Höjer kontrasten till yolo
cv::Mat prepare_image(const cv::Mat& image, bool enabled) {
    if (!enabled) return image;
    cv::Mat lab, blurred;
    cv::cvtColor(image, lab, cv::COLOR_RGB2Lab);
    std::vector<cv::Mat> colors;
    cv::split(lab, colors);
    cv::createCLAHE(3.0, {8, 8})->apply(colors[0], colors[0]);
    cv::GaussianBlur(colors[0], blurred, {}, 1.0);
    cv::addWeighted(colors[0], 1.35, blurred, -0.35, 0, colors[0]);
    cv::merge(colors, lab);
    cv::cvtColor(lab, lab, cv::COLOR_Lab2RGB);
    return lab; //returnerar en bild
}



// Två identiska rutor ger 1. Rutor som inte möts ger 0.
float box_overlap(const cv::Rect2f& first, const cv::Rect2f& second) {
    float intersection = (first & second).area();
    float total_area = first.area() + second.area() - intersection;
    if (total_area <= 0) return 0;
    return intersection / total_area;
}

// Högst poäng får välja först. En överlappande ruta av samma klass tas bort.
std::vector<Detection> remove_duplicate_boxes(std::vector<Detection> boxes, int limit) {
    // Jämför två detektionspoäng för sorteringen.
    std::sort(boxes.begin(), boxes.end(), [](const Detection& first, const Detection& second) { return first.score > second.score; });
    std::vector<Detection> kept;
    for (const Detection& box : boxes) {
        bool duplicate = false;
        for (const Detection& other : kept) {
            if (box.label == other.label && box_overlap(box.box, other.box) > 0.7f) duplicate = true;
        }
        if (!duplicate) kept.push_back(box);
        if (int(kept.size()) >= limit) break;
    }
    return kept;
}
