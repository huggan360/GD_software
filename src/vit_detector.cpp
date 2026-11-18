#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

// Bildförberedelsen finns inuti den exporterade ViT-modellen.
std::vector<float> ImageModel::classify(const cv::Mat& image) {
    c10::InferenceMode inference;
    // Pixelpekaren lånas från OpenCV under modellkörningen; bilden äger fortfarande minnet.
    torch::Tensor input = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kUInt8);
    torch::Tensor scores = network.forward({input}).toTensor().softmax(1).contiguous();
    if (scores.numel() != int64_t(class_names.size())) throw std::runtime_error("ViT:s utdata stämmer inte med klassnamnen.");
    std::vector<float> result(class_names.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] = scores[0][i].item<float>();
    return result;
}

// Hämtar score från yolo och vit där tom lista betyder att det inte går
std::vector<int> match_class_names(const std::vector<std::string>& yolo_names, const std::vector<std::string>& vit_names) {
    if (yolo_names.size() < 2 || yolo_names.size() != vit_names.size()) return {};

    std::map<std::string, int> yolo_index;

    for (size_t i = 0; i < yolo_names.size(); ++i) yolo_index[yolo_names[i]] = int(i);

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
    for (size_t position = 0; position < std::min(size_t(3), scores.size()); ++position) {
        size_t best = 0;
        for (size_t i = 1; i < scores.size(); ++i) if (scores[i] > scores[best]) best = i;
        result.push_back({{"category", names[best]}, {"confidence", scores[best]}});
        scores[best] = -1;
    }
    return result;
}

// Bedömning med vit
void classify_product_vit(ProductTrack& track, ImageModel& vit, const std::vector<int>& class_order, float vit_weight, float minimum_score) {
    std::vector<float> vit_scores(vit.class_names.size(), 0), yolo_scores(vit.class_names.size(), 0);
    bool combine = !class_order.empty();
    track.details["classification_views"] = Json::array();
    for (const ProductPhoto& photo : track.photos) {
        std::vector<float> scores = vit.classify(photo.image); //------------------TILL VIT------------------------------
        for (size_t i = 0; i < scores.size(); ++i) vit_scores[i] += scores[i] / track.photos.size();
        track.details["classification_views"].push_back(photo.time);
        if (!combine) continue;
        if (photo.yolo_scores.size() != class_order.size()) { combine = false; continue; }
        float total = std::accumulate(photo.yolo_scores.begin(), photo.yolo_scores.end(), 0.0f);
        if (!std::isfinite(total) || total <= 0) { combine = false; continue; }
        for (size_t i = 0; i < yolo_scores.size(); ++i) yolo_scores[i] += photo.yolo_scores[class_order[i]] / total / track.photos.size();
    }
    std::vector<float> final_scores = vit_scores;
    if (combine) {
        for (size_t i = 0; i < final_scores.size(); ++i) final_scores[i] = vit_weight * vit_scores[i] + (1 - vit_weight) * yolo_scores[i];
    }
    track.details["vit_top_categories"] = top_three(vit_scores, vit.class_names);
    track.details["top_categories"] = top_three(final_scores, vit.class_names);
    Json best = track.details["top_categories"][0];
    track.details["classification_confidence"] = best["confidence"];
    track.details["category"] = "unknown";
    if (best["confidence"].get<float>() >= minimum_score) track.details["category"] = best["category"];
    track.details["label_source"] = "grocery-vit";
    if (combine) track.details["label_source"] = "yolo-vit";
}
