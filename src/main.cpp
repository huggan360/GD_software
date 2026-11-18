#include "classifier.hpp"
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// läser in json filen
Json read_json(const fs::path& file) {
    std::ifstream input(file);
    if (!input) throw std::runtime_error("Kan inte läsa " + file.string());
    return Json::parse(input);
}

// Byter namn på filen
void write_json(const fs::path& file, const Json& value) {
    auto start = Clock::now();
    std::string text = value.dump(2);
    timings.add("json_encode", start);
    start = Clock::now();
    fs::create_directories(file.parent_path());
    fs::path temporary = file.string() + ".tmp";
    std::ofstream output(temporary);
    output << text << '\n';
    output.close();
    if (!output) throw std::runtime_error("Kan inte skriva " + file.string());
    fs::rename(temporary, file);
    timings.add("json_write", start);
}

// tid för körningen
std::string current_time() {
    std::time_t time = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream text;
    text << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S+00:00");
    return text.str();
}

// Gör sökvägarna relativa till config filen
AppSettings load_settings(const RunOptions& options) {
    AppSettings settings;
    settings.values = YAML::LoadFile(options.config_file.string());
    YAML::Node config = settings.values;
    fs::path config_folder = options.config_file.parent_path();
    settings.model_folder = fs::absolute(config_folder / config["models_directory"].as<std::string>());

    if (!options.model_folder.empty()) settings.model_folder = fs::absolute(options.model_folder);
    settings.backend = config["backend"].as<std::string>();

    if (settings.backend != "hailo" && settings.backend != "torch") throw std::runtime_error("backend ska vara hailo eller torch.");

    settings.vit_file = settings.model_folder / config["classifier"]["checkpoint"].as<std::string>();
    std::string yolo_key = "weights";
    if (settings.backend == "hailo") yolo_key = "hef";
    settings.yolo_file = settings.model_folder / config["detector"][yolo_key].as<std::string>();
    settings.label_file = settings.model_folder / config["detector"]["hef_labels"].as<std::string>("yolo/classes.txt");
    settings.zones_file = fs::absolute(config_folder / config["zones_file"].as<std::string>());
    settings.vit_weight = config["classifier"]["vit_weight"].as<float>(0.7f);
    if (!std::isfinite(settings.vit_weight) || settings.vit_weight < 0 || settings.vit_weight > 1) throw std::runtime_error("vit_weight ska ligga mellan 0 och 1.");

    if (fs::is_regular_file(settings.zones_file)) {
        Json zones = read_json(settings.zones_file);
        for (const auto& camera : zones.items()) {
            if (!config["cameras"][camera.key()]) continue;
            for (const auto& zone : camera.value().items()) config["cameras"][camera.key()][zone.key()] = YAML::Load(zone.value().dump());
        }
    }
    return settings;
}

// läser startargument från orkestrator
RunOptions read_options(int count, char** arguments) {
    RunOptions options;
    options.config_file = fs::canonical("/proc/self/exe").parent_path().parent_path() / "configs/video.yaml";
    for (int i = 1; i < count; ++i) {
        std::string option = arguments[i];
        if (option == "--help") {
            std::cout << "gd_classifier --production|--data-aquisition|--test --record-id ID --request request.json --output result.json [--config video.yaml] [--models-dir mapp] [--managed-retention]\n";
            std::exit(0);
        }
        if (option == "--managed-retention") continue; // Orkestratorn sköter städningen.
        if (option == "--production" || option == "--test" || option == "--data-aquisition" || option == "--data-acquisition") {
            if (!options.mode.empty()) throw std::runtime_error("Välj bara ett körläge.");
            options.mode = "acquisition";
            if (option == "--production") options.mode = "production";
            if (option == "--test") options.mode = "test";
            continue;
        }
        if (i + 1 >= count) throw std::runtime_error("Värde saknas efter " + option);
        std::string value = arguments[++i];
        if (option == "--request") options.request_file = value;
        else if (option == "--output") options.result_file = value;
        else if (option == "--record-id") options.record_id = value;
        else if (option == "--config") options.config_file = value;
        else if (option == "--models-dir") options.model_folder = value;
        else throw std::runtime_error("Okänt startargument: " + option);
    }
    if (options.mode.empty() || options.record_id.empty() || options.request_file.empty() || options.result_file.empty()) throw std::runtime_error("Ange körläge, record-id, request och output.");
    options.result_file = fs::absolute(options.result_file);
    options.config_file = fs::absolute(options.config_file);
    return options;
}

// MAIN
int main(int count, char** arguments) {
    auto program_start = Clock::now();
    try {
        RunOptions options = read_options(count, arguments);
        if (fs::exists(options.result_file)) throw std::runtime_error("Filen finns redan");
        auto start = Clock::now();
        AppSettings settings = load_settings(options);
        timings.add("settings_load", start);
        start = Clock::now();
        Json result = process_video(options, settings);
        timings.add("pipeline_total", start);
        // JSON-delen innehåller tiderna fram till sparandet. Skrivtid och hela programtiden finns i loggen.
        result["timings_ms"] = timings.results();
        result["timing_note"] = "Väggtid i ms. Överlappande steg kan inte summeras. yolo_latency och frame_latency innehåller köväntan. Jämför hastighet med performance.average_frame_ms och processed_fps. JSON-skrivning och hela programtiden finns i loggen.";
        write_json(options.result_file, result);
        timings.add("program_total", program_start);
        timings.print();
        std::cout << options.result_file.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        timings.add("program_total", program_start);
        timings.print();
        std::cerr << "Körningen avbröts: " << error.what() << '\n';
        return 1;
    }
}
