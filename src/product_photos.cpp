#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

//-----------------
//Produktbilder
//--------------------

// plockar ut produktbilder som senare blir använda av ViT
void keep_product_photo(ProductTrack& track, const cv::Mat& image, const cv::Rect2f& box, float score, double time, int limit, const std::vector<float>& yolo_scores) {
    int left = std::clamp(int(std::floor(box.x)), 0, image.cols);
    int top = std::clamp(int(std::floor(box.y)), 0, image.rows);
    int right = std::clamp(int(std::ceil(box.x + box.width)), 0, image.cols);
    int bottom = std::clamp(int(std::ceil(box.y + box.height)), 0, image.rows);
    if (right - left < 2 || bottom - top < 2) return;
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
    int replace_photo = -1;

    // Bilder inom 100ms räknas som samma och vi tar den bästa av de
    if (!track.photos.empty() && time - track.photos.back().time < 0.10) {
        if (quality <= track.photos.back().quality) return;
        replace_photo = int(track.photos.size()) - 1;

    } else if (int(track.photos.size()) >= limit) {

        std::vector<double> times, qualities;
        for (const ProductPhoto& saved : track.photos) {

            times.push_back(saved.time);
            qualities.push_back(saved.quality);
        }
        times.push_back(time);
        qualities.push_back(quality);

        // tar bilder från olika tidpunkter också
        int remove_photo = 0;
        double lowest_value = qualities[0] * 2 * (times[1] - times[0]);
        for (size_t i = 1; i < times.size(); ++i) {
            double time_space = 2 * (times[i] - times[i - 1]);
            if (i + 1 < times.size()) time_space = times[i + 1] - times[i - 1];
            double value = qualities[i] * time_space;
            bool middle_photo = i + 1 < times.size();
            if (value < lowest_value || (value == lowest_value && middle_photo)) {
                lowest_value = value;
                remove_photo = int(i);
            }
        }
        if (remove_photo == int(track.photos.size())) return; // inge bra
        track.photos.erase(track.photos.begin() + remove_photo);
    }

    // Nu har vi bestämt att bilden ska behållas och skapar då produktbilden till ViT.
    cv::Mat photo;
    if (scale < 1) cv::resize(product_area, photo, {}, scale, scale, cv::INTER_AREA);
    else photo = product_area.clone();
    ProductPhoto candidate{quality, time, photo, yolo_scores};
    if (replace_photo >= 0) track.photos[replace_photo] = candidate;
    else track.photos.push_back(candidate);
}

// Används när spåret har produktbilder och vi vill veta vilken som blev tydligast.
double best_photo_time(const ProductTrack& track) {
    size_t best_photo = 0;
    for (size_t i = 1; i < track.photos.size(); ++i) {
        if (track.photos[i].quality > track.photos[best_photo].quality) best_photo = i;
    }
    return track.photos[best_photo].time;
}
