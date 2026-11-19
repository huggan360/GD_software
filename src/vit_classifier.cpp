#include "program.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

//-------------------------------------------------
// 1: Ladda vit och klassnamn
//------------------------------------------------

// Modellfilen innehåller även klassnamn, så vi slipper hålla en separat ViT-lista uppdaterad.
VitClassifier::VitClassifier(const std::filesystem::path& file) {
    at::set_num_threads(4);

    torch::jit::setTensorExprFuserEnabled(false);
    torch::jit::ExtraFilesMap extra_files{{"metadata.json", ""}};
    network = torch::jit::load(file.string(), torch::kCPU, extra_files);
    network.eval();

    Json details = Json::parse(extra_files.at("metadata.json"));

    class_names = details.at("classes").get<std::vector<std::string>>();
}

//-------------------------------------
// 2 Kör en produktbild genom ViT
//---------------------------------------


// returnerar en array med scores
std::vector<float> VitClassifier::classify(const cv::Mat& image) {
    auto stage_start = Clock::now();
    c10::InferenceMode inference;
    torch::Tensor input = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kUInt8);
    timings.add("vit_tensor", stage_start);

    stage_start = Clock::now();
    torch::Tensor logits = network.forward({input}).toTensor();
    timings.add("vit_forward", stage_start);

    stage_start = Clock::now();
    torch::Tensor scores = logits.softmax(1).contiguous();
    timings.add("vit_softmax", stage_start);

    stage_start = Clock::now();
    // Lånar poängen från tensorn och kopierar hela listan på en gång.
    const float* values = scores.data_ptr<float>();
    std::vector<float> result(values, values + class_names.size());
    timings.add("vit_result_copy", stage_start);
    return result;
}
