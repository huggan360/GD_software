
// Följande script skrivet av ChatGPT
// Detta script är för att köra YOLO på en Hailo-8 enhet


#include "program.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <sys/mman.h>
#include <thread>

using namespace hailort;

// Callbacken lämnar bara status och sluttid. Den ändrar aldrig spårning eller tidsstatistik.
struct HailoResult {
    hailo_status status;
    Clock::time_point finished;
};

// En fast bildplats äger sina buffertar tills Hailo är färdig med bilden.
struct HailoRequest {
    float scale = 1;
    int left = 0, top = 0, image_width = 0, image_height = 0;

    Clock::time_point submitted;
    std::shared_ptr<uint8_t> input_buffer;
    std::map<std::string, std::shared_ptr<uint8_t>> output_buffers;
    std::vector<hailort::DmaMappedBuffer> mappings;
    std::unique_ptr<hailort::ConfiguredInferModel::Bindings> bindings;
    std::future<HailoResult> completion;
    hailort::AsyncInferJob job;
};

std::shared_ptr<uint8_t> allocate_hailo_buffer(int bytes) {
    void* address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(address), [bytes](uint8_t* buffer) { munmap(buffer, bytes); });
}

//-------------------
// Steg A: Ladda YOLO på Hailo och förbered bildbuffertarna
//-------------------

YoloClassifier::YoloClassifier(const Settings& settings) {
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

    hailo_vdevice_params_t device_settings;
    hailo_status status = hailo_init_vdevice_params(&device_settings);
    if (status != HAILO_SUCCESS) throw std::runtime_error("Kan inte förbereda Hailo-enheten.");
    device_settings.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;
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
    if (inputs.size() != 1 || outputs.size() != 1) throw std::runtime_error("Använd YOLO11-exporten från GD_tracking.");
    if (outputs[0].format.order != HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS || outputs[0].nms_shape.number_of_classes != class_names.size()) throw std::runtime_error("YOLO:s utgång stämmer inte med klassnamnen.");
    if (inputs[0].shape.features != 3) throw std::runtime_error("YOLO behöver RGB-bilder med tre färgkanaler.");
    input_name = inputs[0].name;
    width = inputs[0].shape.width;
    height = inputs[0].shape.height;
    model->set_batch_size(1);
    model->input(input_name).expect("Bildingången saknas.").set_format_type(HAILO_FORMAT_TYPE_UINT8);
    model->input(input_name).expect("Bildingången saknas.").set_format_order(HAILO_FORMAT_ORDER_NHWC);
    for (const auto& output : outputs) {
        model->output(output.name).expect("Modellutgången saknas.").set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
        model->output(output.name).expect("Modellutgången saknas.").set_format_order(HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS);
    }
    network = model->configure().expect("Kan inte förbereda Hailos asynkrona bildflöde.");
    status = network.activate();
    if (status != HAILO_SUCCESS) throw std::runtime_error("Kan inte aktivera Hailo-nätverket.");
    queue_size = std::min(3, int(network.get_async_queue_size().expect("Kan inte läsa Hailos köstorlek.")));
    if (queue_size == 0) throw std::runtime_error("Hailos bildkö saknar lediga platser.");

    for (int i = 0; i < queue_size; ++i) {
        auto request = std::make_unique<HailoRequest>();
        request->bindings = std::make_unique<ConfiguredInferModel::Bindings>(network.create_bindings().expect("Kan inte koppla bildbuffertarna."));
        int bytes = model->input(input_name).expect("Bildingången saknas.").get_frame_size();
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

YoloClassifier::~YoloClassifier() {
    network.shutdown();
}

//-------------------
// Steg B: Förbered och skicka nästa bild
//-------------------

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
    auto completion = std::make_shared<std::promise<HailoResult>>();
    request.completion = completion->get_future();
    request.submitted = Clock::now();

    request.job = network.run_async(*request.bindings, [completion](const AsyncInferCompletionInfo& info) { completion->set_value({info.status, Clock::now()}); }).expect("Kan inte skicka bilden till Hailo.");
    timings.add("yolo_submit", request.submitted);
    next_send = (next_send + 1) % queue_size;
    ++pending;
}

//-------------------
// Steg C: Hämta modellens svar och läs produktrutorna
//-------------------

std::vector<Detection> YoloClassifier::receive() {
    if (pending == 0) throw std::runtime_error("Det finns ingen väntande YOLO-bild.");
    HailoRequest& request = *requests[next_receive];
    auto stage_start = Clock::now();
    hailo_status status = request.job.wait(std::chrono::seconds(60));
    if (status != HAILO_SUCCESS) throw std::runtime_error("Väntan på YOLO-svaret misslyckades.");
    HailoResult result = request.completion.get();
    timings.add("yolo_wait", stage_start);
    if (result.status != HAILO_SUCCESS) throw std::runtime_error("YOLO-körningen på Hailo misslyckades: " + std::to_string(result.status));
    timings.add("yolo_latency", request.submitted, result.finished);
    float scale = request.scale;
    int left = request.left, top = request.top;
    stage_start = Clock::now();
    std::vector<Detection> detections;
    // YOLO11 lämnar färdiga boxar: först antalet, sedan y1, x1, y2, x2 och poäng.
    const float* values = reinterpret_cast<const float*>(request.output_buffers.at(outputs[0].name).get());
    for (int label = 0; label < int(class_names.size()); ++label) {
        int count = int(*values++);
        for (int i = 0; i < count; ++i, values += 5) {
            if (values[4] < minimum_score) continue;
            float x1 = std::clamp((values[1] * width - left) / scale, 0.0f, float(request.image_width));
            float y1 = std::clamp((values[0] * height - top) / scale, 0.0f, float(request.image_height));
            float x2 = std::clamp((values[3] * width - left) / scale, 0.0f, float(request.image_width));
            float y2 = std::clamp((values[2] * height - top) / scale, 0.0f, float(request.image_height));
            Detection detection;
            detection.box = {x1, y1, x2 - x1, y2 - y1};
            detection.label = label;
            detection.score = values[4];
            detection.class_scores.resize(class_names.size(), 0);
            detection.class_scores[label] = detection.score;
            if (detection.box.area() > 0) detections.push_back(std::move(detection));
        }
    }
    timings.add("yolo_decode", stage_start);
    next_receive = (next_receive + 1) % queue_size;
    --pending;
    stage_start = Clock::now();
    if (int(detections.size()) > max_detections) detections.resize(max_detections);
    timings.add("yolo_limit", stage_start);
    stage_start = Clock::now();
    std::vector<Detection> products;
    for (Detection& detection : detections) {
        const std::string& label = class_names[detection.label];
        if (std::find(excluded_labels.begin(), excluded_labels.end(), label) == excluded_labels.end()) products.push_back(std::move(detection));
    }
    timings.add("detection_filter", stage_start);
    return products;
}
