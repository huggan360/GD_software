#include "program.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <thread>

using namespace hailort;

// Sidjusterat minne kan användas direkt av Hailos DMA utan en extra bildkopia i drivrutinen.
std::shared_ptr<uint8_t> allocate_hailo_buffer(size_t bytes) {
    // Pekaren äger den nya minnesytan tills den lämnas över till shared_ptr nedan.
    void* address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED) throw std::bad_alloc();
    // Deletern får buffertens pekare och frigör minnet när sista ägaren försvinner.
    return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(address), [bytes](uint8_t* buffer) { munmap(buffer, bytes); });
}

//-------------------
// Steg A: Ladda YOLO på Hailo och förbered bildbuffertarna
//-------------------

// Hailos expect avbryter med ett felmeddelande om ett anrop misslyckas.
YoloClassifier::YoloClassifier(const AppSettings& settings) {
    std::ifstream labels(settings.label_file);
    std::string name;
    while (std::getline(labels, name)) {
        if (!name.empty() && name.back() == '\r') name.pop_back();
        if (!name.empty()) class_names.push_back(name);
    }
    if (class_names.empty()) throw std::runtime_error("YOLO behöver klassnamn i yolo/classes.txt.");
    minimum_score = settings.values["detector"]["confidence"].as<float>();
    max_detections = settings.values["detector"]["max_detections"].as<int>();
    excluded_labels = settings.values["detector"]["excluded_labels"].as<std::vector<std::string>>(std::vector<std::string>{});
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
    model = device->create_infer_model(settings.yolo_file.string()).expect("Kan inte läsa YOLO:s HEF-fil.");
    auto inputs = model->hef().get_input_vstream_infos().expect("Kan inte läsa modellens bildingång.");
    outputs = model->hef().get_output_vstream_infos().expect("Kan inte läsa modellens utgångar.");
    if (inputs.size() != 1 || outputs.size() != 6) throw std::runtime_error("Använd YOLO26-exporten med sex utgångar från GD_tracking.");
    if (inputs[0].shape.features != 3) throw std::runtime_error("YOLO behöver RGB-bilder med tre färgkanaler.");
    input_name = inputs[0].name;
    width = inputs[0].shape.width;
    height = inputs[0].shape.height;
    // Batch 1 gör att även sista bilden kan köras, oavsett hur många bilder klippet innehåller.
    model->set_batch_size(1);
    model->input(input_name).expect("Bildingången saknas.").set_format_type(HAILO_FORMAT_TYPE_UINT8);
    model->input(input_name).expect("Bildingången saknas.").set_format_order(HAILO_FORMAT_ORDER_NHWC);
    for (const auto& output : outputs) {
        model->output(output.name).expect("Modellutgången saknas.").set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
        model->output(output.name).expect("Modellutgången saknas.").set_format_order(HAILO_FORMAT_ORDER_NHWC);
    }
    network = model->configure().expect("Kan inte förbereda Hailos asynkrona bildflöde.");
    status = network.activate();
    if (status != HAILO_SUCCESS) throw std::runtime_error("Kan inte aktivera Hailo-nätverket.");
    queue_size = std::min(size_t(3), network.get_async_queue_size().expect("Kan inte läsa Hailos köstorlek."));
    if (queue_size == 0) throw std::runtime_error("Hailos bildkö saknar lediga platser.");

    // Varje bildplats får egna buffertar. Vi återanvänder dem först när bildens svar har hämtats.
    for (size_t i = 0; i < queue_size; ++i) {
        // Den unika pekaren äger bildplatsen och alla dess buffertar.
        auto request = std::make_unique<HailoRequest>();
        request->bindings = std::make_unique<ConfiguredInferModel::Bindings>(network.create_bindings().expect("Kan inte koppla bildbuffertarna."));
        size_t bytes = model->input(input_name).expect("Bildingången saknas.").get_frame_size();
        request->input_buffer = allocate_hailo_buffer(bytes);
        request->mappings.push_back(DmaMappedBuffer::create(*device, request->input_buffer.get(), bytes, HAILO_DMA_BUFFER_DIRECTION_H2D).expect("Kan inte mappa bildbufferten."));
        status = request->bindings->input(input_name).expect("Bildingången saknas.").set_buffer(MemoryView(request->input_buffer.get(), bytes));
        if (status != HAILO_SUCCESS) throw std::runtime_error("Kan inte koppla bildingången.");
        for (const auto& output : outputs) {
            bytes = model->output(output.name).expect("Modellutgången saknas.").get_frame_size();
            request->output_buffers[output.name] = allocate_hailo_buffer(bytes);
            request->mappings.push_back(DmaMappedBuffer::create(*device, request->output_buffers[output.name].get(), bytes, HAILO_DMA_BUFFER_DIRECTION_D2H).expect("Kan inte mappa svarsbufferten."));
            status = request->bindings->output(output.name).expect("Modellutgången saknas.").set_buffer(MemoryView(request->output_buffers[output.name].get(), bytes));
            if (status != HAILO_SUCCESS) throw std::runtime_error("Kan inte koppla modellutgången.");
        }
        requests.push_back(std::move(request));
    }
    timings.add("hailo_setup", stage_start);
}

// Avbryter kvarvarande jobb vid fel och låter dem avslutas innan buffertarna förstörs.
YoloClassifier::~YoloClassifier() {
    network.shutdown();
}

//-------------------
// Steg B: Förbered och skicka nästa bild
//-------------------

// Förbereder och skickar en bild. Hailo fortsätter arbeta efter att funktionen har återvänt.
void YoloClassifier::send(const cv::Mat& image) {
    if (pending >= queue_size) throw std::runtime_error("Hämta ett YOLO-svar innan fler bilder skickas.");
    HailoRequest& request = *requests[next_send];
    auto stage_start = Clock::now();
    cv::Mat padded = resize_for_yolo(image, width, height, request.scale, request.left, request.top);
    request.image_width = image.cols;
    request.image_height = image.rows;
    std::memcpy(request.input_buffer.get(), padded.data, padded.total() * padded.elemSize());
    timings.add("yolo_prepare", stage_start);
    stage_start = Clock::now();
    hailo_status status = network.wait_for_async_ready(std::chrono::seconds(60));
    timings.add("yolo_queue_wait", stage_start);
    if (status != HAILO_SUCCESS) throw std::runtime_error("Hailo kan inte ta emot nästa bild.");
    // Det delade löftet håller callbackens status vid liv tills vi hämtar svaret.
    auto completion = std::make_shared<std::promise<HailoResult>>();
    request.completion = completion->get_future();
    request.submitted = Clock::now();
    // Callbacken lämnar status och sluttid; all spårning och statistik stannar på huvudtråden.
    request.job = network.run_async(*request.bindings, [completion](const AsyncInferCompletionInfo& info) { completion->set_value({info.status, Clock::now()}); }).expect("Kan inte skicka bilden till Hailo.");
    timings.add("yolo_submit", request.submitted);
    next_send = (next_send + 1) % queue_size;
    ++pending;
}

//-------------------
// Steg C: Hämta modellens svar och läs produktrutorna
//-------------------

// Hämtar äldsta bildens svar. Övriga bilder kan fortsätta köras på Hailo under avkodningen.
std::vector<Detection> YoloClassifier::receive() {
    if (pending == 0) throw std::runtime_error("Det finns ingen väntande YOLO-bild.");
    HailoRequest& request = *requests[next_receive];
    auto stage_start = Clock::now();
    hailo_status status = request.job.wait(std::chrono::seconds(60));
    if (status != HAILO_SUCCESS) throw std::runtime_error("Väntan på YOLO-svaret misslyckades.");
    HailoResult result = request.completion.get();
    timings.add("yolo_wait", stage_start);
    if (result.status != HAILO_SUCCESS) throw std::runtime_error("YOLO-körningen på Hailo misslyckades: " + std::to_string(result.status));
    // Latensen innehåller eventuell köväntan på Hailo och ska inte tolkas som ren chipptid.
    timings.add("yolo_latency", request.submitted, result.finished);
    float scale = request.scale;
    int left = request.left, top = request.top;
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
        // Pekarna lånar den här bildplatsens färdiga svar. Buffertarna ägs av request.
        const float* box_values = reinterpret_cast<const float*>(request.output_buffers.at(box_output.name).get());
        // Pekaren lånar klasspoängen från samma bildplats som rutorna.
        const float* class_values = reinterpret_cast<const float*>(request.output_buffers.at(class_output->name).get());
        bool needs_sigmoid = false;
        size_t class_values_count = size_t(class_output->shape.height) * class_output->shape.width * class_names.size();
        for (size_t i = 0; i < class_values_count; ++i) if (class_values[i] < 0 || class_values[i] > 1) needs_sigmoid = true;

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
                const float* edges = box_values + cell * 4;
                float step_x = float(width) / box_output.shape.width;
                float step_y = float(height) / box_output.shape.height;
                float x1 = std::clamp(((x + 0.5f - edges[0]) * step_x - left) / scale, 0.0f, float(request.image_width));
                float y1 = std::clamp(((y + 0.5f - edges[1]) * step_y - top) / scale, 0.0f, float(request.image_height));
                float x2 = std::clamp(((x + 0.5f + edges[2]) * step_x - left) / scale, 0.0f, float(request.image_width));
                float y2 = std::clamp(((y + 0.5f + edges[3]) * step_y - top) / scale, 0.0f, float(request.image_height));
                detection.box = {x1, y1, x2 - x1, y2 - y1};
                if (detection.box.area() > 0) detections.push_back(detection);
            }
        }
    }
    timings.add("yolo_decode", stage_start);
    next_receive = (next_receive + 1) % queue_size;
    --pending;
    stage_start = Clock::now();
    detections = remove_duplicate_boxes(detections, max_detections);
    timings.add("yolo_nms", stage_start);
    stage_start = Clock::now();
    std::vector<Detection> products;
    for (Detection& detection : detections) {
        const std::string& label = class_names[detection.label];
        if (std::find(excluded_labels.begin(), excluded_labels.end(), label) == excluded_labels.end()) products.push_back(std::move(detection));
    }
    timings.add("detection_filter", stage_start);
    return products;
}
