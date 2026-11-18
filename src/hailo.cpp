#include "classifier.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>

using namespace hailort;

// Hailos expect avbryter med ett felmeddelande om ett anrop misslyckas.
HailoDetector::HailoDetector(const AppSettings& settings) : model(Hef::create(settings.yolo_file.string()).expect("Kan inte läsa YOLO:s HEF-fil.")) {
    std::ifstream labels(settings.label_file);
    std::string name;
    while (std::getline(labels, name)) {
        if (!name.empty() && name.back() == '\r') name.pop_back();
        if (!name.empty()) class_names.push_back(name);
    }
    if (class_names.empty()) throw std::runtime_error("YOLO behöver klassnamn i yolo/classes.txt.");
    minimum_score = settings.values["detector"]["confidence"].as<float>();
    class_outputs = settings.values["detector"]["class_outputs"].as<std::vector<std::string>>(std::vector<std::string>{});
    // Fyra klasser ger samma kanalantal som boxutgångarna, så då behöver utgångarna namnges.
    if (class_names.size() == 4 && class_outputs.size() != 3) throw std::runtime_error("Ange klassutgångarnas namn i detector.class_outputs för en HEF med fyra klasser.");

    hailo_vdevice_params_t device_settings;
    hailo_status status = hailo_init_vdevice_params(&device_settings);
    if (status != HAILO_SUCCESS) throw std::runtime_error("Kan inte förbereda Hailo-enheten.");
    device_settings.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;
    // En annan process kan använda Hailo just nu. Vänta högst tre minuter.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
    auto stage_start = Clock::now();
    while (!device) {
        auto available_device = VDevice::create(device_settings);
        if (available_device) device = available_device.release();
        else {
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Hailo-enheten är upptagen eller saknas.");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    timings.add("hailo_wait", stage_start);
    stage_start = Clock::now();
    auto groups = device->configure(model).expect("Kan inte ladda nätverket på Hailo.");
    network = groups.at(0);
    auto inputs = model.get_input_vstream_infos().expect("Kan inte läsa modellens bildingång.");
    outputs = model.get_output_vstream_infos().expect("Kan inte läsa modellens utgångar.");
    if (inputs.size() != 1 || outputs.size() != 6) throw std::runtime_error("Använd YOLO26-exporten med sex utgångar från GD_tracking.");
    input_name = inputs[0].name;
    width = inputs[0].shape.width;
    height = inputs[0].shape.height;
    auto input_settings = network->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).expect("Kan inte skapa bildingången.");
    auto output_settings = network->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).expect("Kan inte skapa modellutgångarna.");
    for (auto& input : input_settings) input.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
    for (auto& output : output_settings) output.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
    streams = std::make_unique<InferVStreams>(InferVStreams::create(*network, input_settings, output_settings).expect("Kan inte öppna Hailos bildflöde."));
    for (const auto& output : outputs) {
        size_t bytes = streams->get_output_by_name(output.name).expect("Modellutgången saknas.").get().get_frame_size();
        output_values[output.name].resize(bytes / sizeof(float));
    }
    activation = network->activate().expect("Kan inte aktivera Hailo-nätverket.");
    timings.add("hailo_setup", stage_start);
}

// Kör modellen och översätter varje rutnätspunkt till en produktruta i den inskickade bilden.
std::vector<Detection> HailoDetector::detect(const cv::Mat& image) {
    auto stage_start = Clock::now();
    float scale;
    int left, top;
    cv::Mat padded = resize_for_yolo(image, width, height, scale, left, top);
    // MemoryView lånar bildens och buffertarnas datapekare under Hailo-körningen.
    std::map<std::string, MemoryView> inputs, results;
    inputs.emplace(input_name, MemoryView(padded.data, padded.total() * padded.elemSize()));
    for (auto& output : output_values) results.emplace(output.first, MemoryView(output.second.data(), output.second.size() * sizeof(float)));
    timings.add("yolo_prepare", stage_start);
    stage_start = Clock::now();
    hailo_status status = streams->infer(inputs, results, 1);
    // Anropet väntar på hela svaret. Tiden innehåller överföring till och från Hailo.
    timings.add("yolo_inference", stage_start);
    if (status != HAILO_SUCCESS) throw std::runtime_error("YOLO-körningen på Hailo misslyckades: " + std::to_string(status));

    stage_start = Clock::now();
    std::vector<Detection> detections;
    for (const auto& box_output : outputs) {
        bool is_class_output = std::find(class_outputs.begin(), class_outputs.end(), box_output.name) != class_outputs.end();
        if (box_output.shape.features != 4 || is_class_output) continue;
        // Pekaren lånar beskrivningen av klassutgången från outputs. Den ska inte frigöras här.
        const hailo_vstream_info_t* class_output = nullptr;
        for (const auto& output : outputs) {
            bool correct_name = class_outputs.empty() || std::find(class_outputs.begin(), class_outputs.end(), output.name) != class_outputs.end();
            bool same_grid = output.shape.height == box_output.shape.height && output.shape.width == box_output.shape.width;
            if (std::string(output.name) != box_output.name && same_grid && output.shape.features == class_names.size() && correct_name) class_output = &output;
        }
        if (!class_output) throw std::runtime_error("HEF-modellens klassutgångar stämmer inte med klassnamnen.");
        const std::vector<float>& box_values = output_values.at(box_output.name);
        const std::vector<float>& class_values = output_values.at(class_output->name);
        bool needs_sigmoid = false;
        for (float value : class_values) if (value < 0 || value > 1) needs_sigmoid = true;

        for (unsigned y = 0; y < box_output.shape.height; ++y) {
            for (unsigned x = 0; x < box_output.shape.width; ++x) {
                size_t cell = y * box_output.shape.width + x;
                Detection detection;
                detection.score = 0;
                detection.label = 0;
                detection.class_scores.resize(class_names.size());
                for (size_t label = 0; label < class_names.size(); ++label) {
                    float score = class_values[cell * class_names.size() + label];
                    if (needs_sigmoid) score = 1.0f / (1.0f + std::exp(-score));
                    detection.class_scores[label] = score;
                    if (score > detection.score) { detection.score = score; detection.label = int(label); }
                }
                if (detection.score < minimum_score) continue;
                // Pekaren lånar cellens fyra kantavstånd ur box_values.
                const float* edges = box_values.data() + cell * 4;
                float step_x = float(width) / box_output.shape.width;
                float step_y = float(height) / box_output.shape.height;
                float x1 = std::clamp(((x + 0.5f - edges[0]) * step_x - left) / scale, 0.0f, float(image.cols));
                float y1 = std::clamp(((y + 0.5f - edges[1]) * step_y - top) / scale, 0.0f, float(image.rows));
                float x2 = std::clamp(((x + 0.5f + edges[2]) * step_x - left) / scale, 0.0f, float(image.cols));
                float y2 = std::clamp(((y + 0.5f + edges[3]) * step_y - top) / scale, 0.0f, float(image.rows));
                detection.box = {x1, y1, x2 - x1, y2 - y1};
                if (detection.box.area() > 0) detections.push_back(detection);
            }
        }
    }
    timings.add("yolo_decode", stage_start);
    return detections;
}
