#include "program.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>


//---------------------------------------
// Förbered bilder och jämför produktrutor
//---------------------------------------------------

// Anpassar bildstorleken för att bevara proportionen
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

// Kollar överlapp mellan två boundry boxes
// Intersection over Union (IoU)
// Källa: https://www.ultralytics.com/glossary/intersection-over-union-iou
float box_overlap(const cv::Rect2f& first, const cv::Rect2f& second) {
    float intersection = (first & second).area();
    float total_area = first.area() + second.area() - intersection;
    if (total_area <= 0) return 0;
    return intersection / total_area;
}
