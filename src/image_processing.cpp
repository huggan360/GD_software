#include "program.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>


//-------------------
// Förbered bilder och jämför produktrutor
//-------------------

// Anpassar bildstorleken och fyller ut kanterna så att proportionerna behålls.
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



//
float box_overlap(const cv::Rect2f& first, const cv::Rect2f& second) {
    float intersection = (first & second).area();
    float total_area = first.area() + second.area() - intersection;
    if (total_area <= 0) return 0;
    return intersection / total_area;
}

// ta bort boxar som överlappar
std::vector<Detection> remove_duplicate_boxes(std::vector<Detection> boxes, int limit) {
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
