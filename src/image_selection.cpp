#include "program.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

//-------------------
// Välj produktbilder till ViT
//-------------------

// Behåller upp till fem tydliga utsnitt i minnet, spridda över objektets tid i videon.
void keep_product_photo(ProductTrack& track, const cv::Mat& image, const cv::Rect2f& box, float score, double time, int limit, const std::vector<float>& yolo_scores) {
    auto stage_start = Clock::now();
    int left = std::clamp(int(std::floor(box.x)), 0, image.cols);
    int top = std::clamp(int(std::floor(box.y)), 0, image.rows);
    int right = std::clamp(int(std::ceil(box.x + box.width)), 0, image.cols);
    int bottom = std::clamp(int(std::ceil(box.y + box.height)), 0, image.rows);
    if (right - left < 2 || bottom - top < 2) { timings.add("photo_quality", stage_start); return; }
    // Detta är bara en vy in i bildrutan, inte en sparad kopia av produkten.
    cv::Mat product_area = image(cv::Rect(left, top, right - left, bottom - top));
    double scale = std::min(1.0, 384.0 / std::max(product_area.cols, product_area.rows));
    cv::Mat grey, edges;
    cv::cvtColor(product_area, grey, cv::COLOR_RGB2GRAY);
    if (scale < 1) cv::resize(grey, grey, {}, scale, scale, cv::INTER_AREA);
    cv::Laplacian(grey, edges, CV_64F);
    cv::Scalar average, deviation;
    cv::meanStdDev(edges, average, deviation);
    double sharpness = deviation[0] * deviation[0];
    double quality = score * std::sqrt(grey.total()) * std::log1p(sharpness);
    timings.add("photo_quality", stage_start);
    stage_start = Clock::now();
    int replace_photo = -1;

    // Bilder inom 100ms räknas som samma och vi tar den bästa av de
    for (size_t i = 0; i < track.photos.size(); ++i) {
        if (std::abs(time - track.photos[i].time) < 0.10) { replace_photo = int(i); break; }
    }
    // tar bilder från olika tidpunkter också
    if (replace_photo < 0 && int(track.photos.size()) >= limit) {
        replace_photo = int(std::min_element(track.photos.begin(), track.photos.end(), [](const ProductPhoto& a, const ProductPhoto& b) { return a.quality < b.quality; }) - track.photos.begin());
    }
    if (replace_photo >= 0 && quality <= track.photos[replace_photo].quality) { timings.add("photo_selection", stage_start); return; }

    timings.add("photo_selection", stage_start);
    stage_start = Clock::now();
    // Nu har vi bestämt att bilden ska behållas och skapar då produktbilden till ViT.
    cv::Mat photo;
    if (scale < 1) cv::resize(product_area, photo, {}, scale, scale, cv::INTER_AREA);
    else photo = product_area.clone();
    ProductPhoto candidate{quality, time, photo, yolo_scores};
    if (replace_photo >= 0) track.photos[replace_photo] = candidate;
    else track.photos.push_back(candidate);
    timings.add("photo_save", stage_start);
}

// Samlar objektets klassificering, tid och en produktruta i svaret.
Json product_result(const ProductTrack& track) {
    double time = track.observations.front().time;
    if (!track.photos.empty()) {
        const auto best = std::max_element(track.photos.begin(), track.photos.end(), [](const ProductPhoto& a, const ProductPhoto& b) { return a.quality < b.quality; });
        time = best->time;
    }
    const auto best = std::min_element(track.observations.begin(), track.observations.end(), [time](const Observation& a, const Observation& b) { return std::abs(a.time - time) < std::abs(b.time - time); });
    Json result = track.details;
    result["start"] = track.observations.front().time;
    result["end"] = track.observations.back().time;
    result["time"] = best->time;
    result["box"] = {best->box.x, best->box.y, best->box.x + best->box.width, best->box.y + best->box.height};
    return result;
}
