#include "classifier.hpp"
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>

Json read_json(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    return Json::parse(input);
}

void write_json(const fs::path& path, const Json& value) {
    fs::create_directories(path.parent_path());
    fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary);
    output << value.dump(2) << '\n';
    output.close();
    if (!output) throw std::runtime_error("Cannot write " + temporary.string());
    fs::rename(temporary, path);
}

std::string now() {
    std::time_t time = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream result;
    result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S+00:00");
    return result.str();
}

Settings load_settings(const Options& options) {
    Settings settings;
    settings.config = YAML::LoadFile(options.config.string());
    auto config = settings.config;
    auto relative = [&](const std::string& name) { return fs::absolute(options.config.parent_path() / config[name].as<std::string>()).lexically_normal(); };
    settings.model_root = options.models.empty() ? relative("models_directory") : fs::absolute(options.models);
    settings.backend = config["backend"].as<std::string>();
    if (settings.backend != "hailo" && settings.backend != "torch") throw std::runtime_error("backend must be hailo or torch.");
    settings.vit = settings.model_root / config["classifier"]["checkpoint"].as<std::string>("vit/vit.pth");
    settings.yolo = settings.model_root / config["detector"][settings.backend == "hailo" ? "hef" : "weights"].as<std::string>();
    settings.labels = settings.model_root / config["detector"]["hef_labels"].as<std::string>("yolo/classes.txt");
    settings.zones_file = relative("zones_file");
    settings.acquisition = relative("acquisition_directory");
    if (fs::is_regular_file(settings.zones_file)) {
        auto saved = read_json(settings.zones_file);
        for (const auto& camera : saved.items()) if (config["cameras"][camera.key()]) for (const auto& zone : camera.value().items()) config["cameras"][camera.key()][zone.key()] = YAML::Load(zone.value().dump());
    }
    const auto tracking = config["tracking"];
    double sample_fps = tracking["sample_fps"].as<double>();
    if (!std::isfinite(sample_fps) || sample_fps <= 0 || sample_fps > 60 || tracking["stable_frames"].as<int>() < 1 || tracking["dwell_seconds"].as<double>() < 0 || tracking["max_gap_seconds"].as<double>() <= 0 || tracking["max_tracks"].as<int>() < 1 || config["classifier"]["crops_per_track"].as<int>() < 1) throw std::runtime_error("Invalid tracking settings.");
    return settings;
}

// These flags are the existing smart_fridge_edge handoff, not training settings.
Options read_options(int argc, char** argv) {
    Options options;
    const fs::path root = fs::canonical("/proc/self/exe").parent_path().parent_path();
    options.config = root / "configs/video.yaml";
    for (int index = 1; index < argc; ++index) {
        std::string flag = argv[index];
        if (flag == "--help") { std::cout << "gd_classifier --production|--data-aquisition|--test --record-id ID --request request.json --output result.json [--config video.yaml] [--models-dir directory] [--managed-retention]\n"; std::exit(0); }
        if (flag == "--production" || flag == "--test" || flag == "--data-aquisition" || flag == "--data-acquisition") {
            if (!options.mode.empty()) throw std::runtime_error("Choose one run mode.");
            options.mode = flag == "--production" ? "production" : flag == "--test" ? "test" : "acquisition";
        } else if (flag == "--managed-retention") options.managed_retention = true;
        else {
            if (index + 1 == argc) throw std::runtime_error("Missing value for " + flag);
            std::string value = argv[++index];
            if (flag == "--request") options.request = value;
            else if (flag == "--output") options.output = value;
            else if (flag == "--record-id") options.record_id = value;
            else if (flag == "--config") options.config = value;
            else if (flag == "--models-dir") options.models = value;
            else throw std::runtime_error("Unknown option: " + flag);
        }
    }
    if (options.mode.empty() || options.record_id.empty() || options.request.empty() || options.output.empty()) throw std::runtime_error("A mode, --record-id, --request and --output are required.");
    options.output = fs::absolute(options.output);
    options.config = fs::absolute(options.config);
    return options;
}

int main(int argc, char** argv) {
    try {
        const auto options = read_options(argc, argv);
        if (fs::exists(options.output)) throw std::runtime_error("Choose a new result path; existing results are not overwritten.");
        auto settings = load_settings(options);
        auto result = run_clip(options, settings);
        // The reviewer needs a local playable copy; production reads the source directly.
        if (options.mode == "test") {
            std::random_device random;
            std::ostringstream identity;
            for (int index = 0; index < 3; ++index) identity << std::hex << std::setw(8) << std::setfill('0') << uint32_t(random());
            std::string id = identity.str();
            auto library = settings.acquisition / "videos";
            auto folder = library / (".publish-" + id);
            fs::create_directories(folder);
            std::string original = "original" + fs::path(result.at("source").get<std::string>()).extension().string();
            fs::copy_file(result.at("source").get<std::string>(), folder / original);
            fs::copy_file(result.at("source").get<std::string>(), folder / "video.mp4");
            for (const auto& file : fs::directory_iterator(options.output.parent_path())) if (file.path().filename() == "poster.jpg" || file.path().filename().string().rfind("track-", 0) == 0) fs::copy_file(file.path(), folder / file.path().filename());
            write_json(folder / "result.json", result);
            write_json(folder / "meta.json", {{"id", id}, {"filename", result["event_id"].get<std::string>() + ".mp4"}, {"camera", result["camera"]}, {"event_id", result["event_id"]}, {"created", std::time(nullptr)}, {"editor", "live system (--test)"}, {"status", "done"}, {"progress", 100}, {"message", "C++ inference"}, {"original", original}});
            fs::rename(folder, library / id);
        }
        write_json(options.output, result);
        std::cout << options.output.string() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "Classification failed: " << error.what() << '\n'; return 1; }
}
