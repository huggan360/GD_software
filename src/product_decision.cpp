#include "program.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

// Hämtar score från yolo och vit där tom lista betyder att det inte går
std::vector<int> match_class_names(const std::vector<std::string>& yolo_names, const std::vector<std::string>& vit_names) {
    if (yolo_names.size() < 2 || yolo_names.size() != vit_names.size()) return {};

    std::map<std::string, int> yolo_index;

    for (int i = 0; i < int(yolo_names.size()); ++i) yolo_index[yolo_names[i]] = i;

    if (yolo_index.size() != yolo_names.size()) return {};

    std::vector<int> order;

    for (const std::string& name : vit_names) {
        auto found = yolo_index.find(name);

        if (found == yolo_index.end()) return {};

        order.push_back(found->second);
    }
    return order;
}

// Plockar ut de tre största poängen
Json top_three(std::vector<float> scores, const std::vector<std::string>& names) {
    Json result = Json::array();
    for (int position = 0; position < std::min(3, int(scores.size())); ++position) {
        int best = 0;
        for (int i = 1; i < int(scores.size()); ++i) if (scores[i] > scores[best]) best = i;
        result.push_back({{"category", names[best]}, {"confidence", scores[best]}});
        scores[best] = -1;
    }
    return result;
}

// Sparar YOLO:s egna förslag från samma bilder som valts till ViT.
void save_yolo_predictions(ProductTrack& track, const std::vector<std::string>& class_names) {
    auto start = Clock::now();
    std::vector<float> scores(class_names.size(), 0);
    int photos = 0;
    for (const ProductPhoto& photo : track.photos) {
        if (photo.yolo_scores.size() != scores.size()) continue;
        for (int i = 0; i < int(scores.size()); ++i) scores[i] += photo.yolo_scores[i];
        ++photos;
    }
    track.details["yolo_top_categories"] = Json::array();
    if (photos > 0) {
        for (float& score : scores) score /= photos;
        track.details["yolo_top_categories"] = top_three(scores, class_names);
    }
    timings.add("yolo_scores", start);
}

// Väger modellernas förslag och bestämmer produktnamnet för hela objektspåret.
void decide_product(ProductTrack& track, VitClassifier& vit, const std::vector<int>& class_order, float vit_weight, float minimum_score) {
    std::vector<float> vit_scores(vit.class_names.size(), 0), yolo_scores(vit.class_names.size(), 0);
    bool combine = !class_order.empty();
    for (const ProductPhoto& photo : track.photos) {
        auto image_start = Clock::now();
        std::vector<float> scores = vit.classify(photo.image); //------------------TILL VIT------------------------------
        timings.add("vit_image_total", image_start);
        for (int i = 0; i < int(scores.size()); ++i) vit_scores[i] += scores[i] / track.photos.size();
        if (!combine) continue;
        if (photo.yolo_scores.size() != class_order.size()) { combine = false; continue; }
        float total = std::accumulate(photo.yolo_scores.begin(), photo.yolo_scores.end(), 0.0f);
        if (!std::isfinite(total) || total <= 0) { combine = false; continue; }
        for (int i = 0; i < int(yolo_scores.size()); ++i) yolo_scores[i] += photo.yolo_scores[class_order[i]] / total / track.photos.size();
    }
    auto decision_start = Clock::now();
    std::vector<float> final_scores = vit_scores;
    if (combine) {
        for (int i = 0; i < int(final_scores.size()); ++i) final_scores[i] = vit_weight * vit_scores[i] + (1 - vit_weight) * yolo_scores[i];
    }
    track.details["vit_top_categories"] = top_three(vit_scores, vit.class_names);
    Json best = top_three(final_scores, vit.class_names)[0];
    track.details["classification_confidence"] = best["confidence"];
    track.details["category"] = "unknown";
    if (best["confidence"].get<float>() >= minimum_score) track.details["category"] = best["category"];
    timings.add("vit_decision", decision_start);
}
