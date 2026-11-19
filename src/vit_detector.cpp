#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

// Bildförberedelsen finns inuti den exporterade ViT-modellen.
std::vector<float> VitModel::classify(const cv::Mat& image) {
    auto stage_start = Clock::now();
    c10::InferenceMode inference;
    // Pixelpekaren lånas från OpenCV under modellkörningen; bilden äger fortfarande minnet.
    torch::Tensor input = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kUInt8);
    double tensor_ms = timings.add("vit_tensor", stage_start);
    stage_start = Clock::now();
    // Den exporterade modellen innehåller både RGB-förberedelsen och själva ViT.
    torch::Tensor logits = network.forward({input}).toTensor();
    double forward_ms = timings.add("vit_forward", stage_start);
    stage_start = Clock::now();
    torch::Tensor scores = logits.softmax(1).contiguous();
    double softmax_ms = timings.add("vit_softmax", stage_start);
    if (scores.numel() != int64_t(class_names.size())) throw std::runtime_error("ViT:s utdata stämmer inte med klassnamnen.");
    stage_start = Clock::now();
    std::vector<float> result(class_names.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] = scores[0][i].item<float>();
    double copy_ms = timings.add("vit_result_copy", stage_start);
    std::cout << "Tid ViT-bild (ms): tensor=" << tensor_ms << ", forward=" << forward_ms << ", softmax=" << softmax_ms << ", result_copy=" << copy_ms << std::endl;
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
void classify_product_vit(ProductTrack& track, VitModel& vit, const std::vector<int>& class_order, float vit_weight, float minimum_score) {
    std::vector<float> vit_scores(vit.class_names.size(), 0), yolo_scores(vit.class_names.size(), 0);
    bool combine = !class_order.empty();
    track.details["classification_views"] = Json::array();
    track.details["classification_image_times_ms"] = Json::array();
    int completed_photos = 0;
    for (const ProductPhoto& photo : track.photos) {
        auto image_start = Clock::now();
        std::vector<float> scores = vit.classify(photo.image); //------------------TILL VIT------------------------------
        double image_ms = timings.add("vit_image_total", image_start);
        track.details["classification_image_times_ms"].push_back({{"time_seconds", photo.time}, {"total_ms", image_ms}});
        ++completed_photos;
        std::cout << "ViT objekt " << track.details["id"].get<std::string>() << ": bild " << completed_photos << " av " << track.photos.size() << " klar." << std::endl;
        for (size_t i = 0; i < scores.size(); ++i) vit_scores[i] += scores[i] / track.photos.size();
        track.details["classification_views"].push_back(photo.time);
        if (!combine) continue;
        if (photo.yolo_scores.size() != class_order.size()) { combine = false; continue; }
        float total = std::accumulate(photo.yolo_scores.begin(), photo.yolo_scores.end(), 0.0f);
        if (!std::isfinite(total) || total <= 0) { combine = false; continue; }
        for (size_t i = 0; i < yolo_scores.size(); ++i) yolo_scores[i] += photo.yolo_scores[class_order[i]] / total / track.photos.size();
    }
    auto decision_start = Clock::now();
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
    timings.add("vit_decision", decision_start);
}
