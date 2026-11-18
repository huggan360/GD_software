#include "classifier.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <algorithm>

// Modellfilen innehåller även klassnamn, så vi slipper hålla en separat ViT-lista uppdaterad.
ImageModel::ImageModel(const fs::path& file, const std::string& expected_format) {
    at::set_num_threads(4);
    // Kör modellens beräkningar direkt. CPU-fusionens kompilering kan fastna länge på Pi:n.
    torch::jit::setTensorExprFuserEnabled(false);
    torch::jit::ExtraFilesMap extra_files{{"metadata.json", ""}};
    network = torch::jit::load(file.string(), torch::kCPU, extra_files);
    network.eval();
    Json details = Json::parse(extra_files.at("metadata.json"));
    if (details.at("format") != expected_format) throw std::runtime_error("Fel modellformat. Exportera modellen från rätt GD-repo.");
    class_names = details.at("classes").get<std::vector<std::string>>();
    if (class_names.empty()) throw std::runtime_error("Modellen saknar klassnamn.");
    image_size = details.value("image_size", 640);
}
