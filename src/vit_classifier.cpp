#include "program.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <iostream>

//-------------------
// Steg A: Ladda ViT-modellen och dess klassnamn
//-------------------

// Modellfilen innehåller även klassnamn, så vi slipper hålla en separat ViT-lista uppdaterad.
VitClassifier::VitClassifier(const fs::path& file) {
    at::set_num_threads(4);
    // Kör modellens beräkningar direkt. CPU-fusionens kompilering kan fastna länge på Pi:n.
    torch::jit::setTensorExprFuserEnabled(false);
    torch::jit::ExtraFilesMap extra_files{{"metadata.json", ""}};
    network = torch::jit::load(file.string(), torch::kCPU, extra_files);
    network.eval();
    Json details = Json::parse(extra_files.at("metadata.json"));
    if (details.at("format") != "gd-vit-rgb-v1") throw std::runtime_error("Fel ViT-format. Exportera modellen från GD_vision.");
    class_names = details.at("classes").get<std::vector<std::string>>();
    if (class_names.empty()) throw std::runtime_error("Modellen saknar klassnamn.");
}

//-------------------
// Steg B: Kör en produktbild genom ViT
//-------------------

// Bildförberedelsen finns inuti den exporterade ViT-modellen.
std::vector<float> VitClassifier::classify(const cv::Mat& image) {
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
