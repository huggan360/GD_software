#pragma once

#include <cstdint>
#include <chrono>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

// inte standard
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <ByteTrack/BYTETracker.h>
#include <torch/script.h>
#include <yaml-cpp/yaml.h>
#include <hailo/hailort.hpp>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;



// De sista 300 pixlarna till höger skickas aldrig till YOLO
const int IGNORE_RIGHT_PIXELS = 300;


// En tidsmätning i millisekunder, samlad över alla anrop till samma steg.
struct TimeReading {
    size_t calls = 0;
    double total = 0;
};

// Samlar väggtid för den här körningen. Tider för hela steg innehåller även deras delsteg.
class Timings {
    public:
        double add(const std::string& name, Clock::time_point start, Clock::time_point finish = Clock::now());
        void print() const;

    private:
        std::map<std::string, TimeReading> readings;

};

extern Timings timings;

// En ruta som YOLO hittat, tillsammans med modellens klasspoäng.
struct Detection {
    cv::Rect2f box;
    float score;
    int label;
    std::vector<float> class_scores;
};

// produktbild till vit
struct ProductPhoto {
    double quality, time;
    cv::Mat image;
    std::vector<float> yolo_scores;
};

// håller samma föremål i klippet
struct Observation {
    double time;
    cv::Rect2f box;
};

struct ProductTrack {
    Json details;
    std::vector<ProductPhoto> photos;
    std::vector<Observation> observations;
    Json events = Json::array();
};

// Ett objekt som ByteTrack har följt i den aktuella bildrutan.
struct TrackedObject {
    size_t id;
    cv::Rect2f box;
    float score;
    std::vector<float> yolo_scores;
};

// Det som orkestratorn skickar när programmet startas.
struct RunOptions {
    std::filesystem::path request_file, result_file, config_file, model_folder;
    std::string record_id, mode;
};

// filer och inställningar som används
struct Settings {
    YAML::Node values;
    std::filesystem::path model_folder, vit_file, yolo_file, label_file, zones_file;
    float vit_weight = 0.7f;
};


struct HailoRequest;


// Kör ViT på cpu.
class VitClassifier {
public:
    VitClassifier(const std::filesystem::path& file);

    std::vector<float> classify(const cv::Mat& image);
    std::vector<std::string> class_names;
private:
    torch::jit::Module network;
};


// Här finns de Hailo-specifika delarna. Resten av programmet arbetar med vanliga produktrutor.
class YoloClassifier {
public:
    YoloClassifier(const Settings& settings);
    ~YoloClassifier();

    void send(const cv::Mat& image);
    std::vector<Detection> receive();
    std::vector<std::string> class_names;
    size_t queue_size = 1;

private:
    std::unique_ptr<hailort::VDevice> device;
    std::shared_ptr<hailort::InferModel> model;
    hailort::ConfiguredInferModel network;
    std::vector<std::unique_ptr<HailoRequest>> requests;
    size_t next_send = 0, next_receive = 0, pending = 0;
    std::vector<hailo_vstream_info_t> outputs;
    std::vector<std::string> class_outputs;
    std::string input_name;
    int width = 640, height = 640;
    float minimum_score = 0.15f;
    int max_detections = 100;
    std::vector<std::string> excluded_labels;
};

// bytetrack tracker
class Tracker {
public:
    Tracker(double sample_fps, const YAML::Node& settings, cv::Size image_size);
    std::vector<TrackedObject> update(const std::vector<Detection>& boxes, double time);
    std::map<size_t, ProductTrack> products;
private:
    byte_track::BYTETracker tracker;
    cv::Size image_size;
};


//-------------------------------------------------
// Hjälpfunktioner
//------------------------------------------------

// Läser en JSON-fil.
Json read_json(const std::filesystem::path& file);

// Skriver JSON via en tillfällig fil så att orkestratorn får ett komplett svar.
void write_json(const std::filesystem::path& file, const Json& value);

// Läser YAML-filen och hittar modellerna och kamerornas zoner.
Settings load_settings(const RunOptions& options);

// Anpassar en bild till modellens storlek med utfyllda kanter.
cv::Mat resize_for_yolo(const cv::Mat& image, int width, int height, float& scale, int& left, int& top);

// Räknar ut hur stor del av två rutor som överlappar, mellan 0 och 1.
float box_overlap(const cv::Rect2f& first, const cv::Rect2f& second);

// Behåller de tydligaste rutorna och tar bort dubbletter.
std::vector<Detection> remove_duplicate_boxes(std::vector<Detection> boxes, int limit);

// Analyserar ett klipp och bygger svaret till orkestratorn.
Json process_video(const RunOptions& options, const Settings& settings);

// Matchar klassnamnen så att YOLO:s poäng hamnar i samma ordning som ViT:s.
std::vector<int> match_class_names(const std::vector<std::string>& yolo_names, const std::vector<std::string>& vit_names);

// Sparar YOLO:s förslag från objektets utvalda produktbilder.
void save_yolo_predictions(ProductTrack& track, const std::vector<std::string>& class_names);

// Klassificerar objektets bilder med ViT och väger samman användbara modellförslag.
void decide_product(ProductTrack& track, VitClassifier& vit, const std::vector<int>& class_order, float vit_weight, float minimum_score);

// Väljer tydliga produktbilder från olika tidpunkter till ViT.
void keep_product_photo(ProductTrack& track, const cv::Mat& image, const cv::Rect2f& box, float score, double time, int limit, const std::vector<float>& yolo_scores);

// Samlar objektets klassificering, tid och en produktruta i svaret.
Json product_result(const ProductTrack& track);

// Läser kamerans zoner för insida och utsida.
Json read_zones(const YAML::Node& camera);

// Hittar in- och utpassager längs ett produktspår.
void find_movements(ProductTrack& track, const Json& zones, const YAML::Node& settings, double offset);

// Beräknar hur långt objektets mittpunkt har förflyttat sig i bilden.
double track_movement(const std::vector<Observation>& observations);




