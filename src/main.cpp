#include "program.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

//-------------------
// Läs och skriv JSON-filer
//-------------------

// Läser in en json
Json read_json(const std::filesystem::path& file) {
    std::ifstream input(file);
    if (!input) throw std::runtime_error("Kan inte läsa " + file.string());
    return Json::parse(input);
}

// Skriver klart en tillfällig fil och ger den sedan resultatfilens namn.

void write_json(const std::filesystem::path& file, const Json& value) {
    std::filesystem::create_directories(file.parent_path());
    std::filesystem::path temporary = file.string() + ".tmp";
    std::ofstream output(temporary);
    output << value.dump() << '\n';
    output.close();

    if (!output) throw std::runtime_error("Kan inte skriva " + file.string());
    std::filesystem::rename(temporary, file);
}

// Gör sökvägarna relativa till config filen
Settings load_settings(const RunOptions& options) {

    Settings settings;
    settings.values = YAML::LoadFile(options.config_file.string());
    YAML::Node config = settings.values;
    std::filesystem::path config_folder = options.config_file.parent_path();
    settings.model_folder = std::filesystem::absolute(config_folder / config["models_directory"].as<std::string>());

    if (!options.model_folder.empty()) settings.model_folder = std::filesystem::absolute(options.model_folder);
    settings.vit_file = settings.model_folder / config["classifier"]["checkpoint"].as<std::string>();
    settings.yolo_file = settings.model_folder / config["detector"]["hef"].as<std::string>();
    settings.label_file = settings.model_folder / config["detector"]["hef_labels"].as<std::string>("yolo/classes.txt");
    settings.zones_file = std::filesystem::absolute(config_folder / config["zones_file"].as<std::string>());
    settings.vit_weight = config["classifier"]["vit_weight"].as<float>(0.7f);
    if (!std::isfinite(settings.vit_weight) || settings.vit_weight < 0 || settings.vit_weight > 1) throw std::runtime_error("vit_weight ska ligga mellan 0 och 1.");

    if (std::filesystem::is_regular_file(settings.zones_file)) {
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

    options.config_file = std::filesystem::canonical("/proc/self/exe").parent_path().parent_path() / "configs/video.yaml";

    for (int i = 1; i < count; i++) {
        std::string arg = arguments[i];

        if (arg == "--help") {
            std::cout << "Du kör " "< --production | --data-aquisition | --test > " "--record-id ID --request FIL --output FIL\n"; std::exit(0);
        }

        if (arg == "--managed-retention")
            continue;

        if (arg == "--production" || arg == "--test" ||
            arg == "--data-aquisition" || arg == "--data-acquisition") {

            if (!options.mode.empty())
                throw std::runtime_error("Välj bara ett körläge.");

            if (arg == "--production")
                options.mode = "production";
            else if (arg == "--test")
                options.mode = "test";
            else
                options.mode = "acquisition";

            continue;
            }

            if (i + 1 >= count)
                throw std::runtime_error("missing value");  //efter mode vill vi ha argument plus ett värde

        std::string value = arguments[++i];

        if (arg == "--request")
            options.request_file = value;
        else if (arg == "--output")
            options.result_file = value;
        else if (arg == "--record-id")
            options.record_id = value;
        else if (arg == "--config")
            options.config_file = value;
        else if (arg == "--models-dir")
            options.model_folder = value;
        else
            throw std::runtime_error("Unkown argument");
    }

    if (options.mode.empty() || options.record_id.empty() || options.request_file.empty() || options.result_file.empty()) {
        throw std::runtime_error("Enter correct arguments");
        }

    options.result_file = std::filesystem::absolute(options.result_file);
    options.config_file = std::filesystem::absolute(options.config_file);

    return options;
}

//-------------------
// Programmets startpunkt
//-------------------

// main kör pipeline
int main(int count, char** arguments) {

    auto program_start = Clock::now();

    try {

        //-------------------------------------
        // 1: läs argumenten och inställningarna
        //--------------------------------------

        RunOptions options = read_options(count, arguments);

        auto start = Clock::now();
        Settings settings = load_settings(options);
        timings.add("settings_load", start);

        //-----------------------------------------
        // 2: Kör videon genom pipelinen
        //---------------------------------------

        start = Clock::now();
        Json result = process_video(options, settings);
        timings.add("pipeline_total", start);

        //---------------------------------------------
        //3: Spara resultatet och printa -tidsmätningarna
        //--------------------------------------------

        start = Clock::now();
        write_json(options.result_file, result);
        timings.add("json_write", start);
        timings.add("program_total", program_start);
        timings.print();

        return 0;
    } catch (const std::exception&) {


        std::cerr << "Kunde inte köras" << '\n';
        return 1;
    }
}
