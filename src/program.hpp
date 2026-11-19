#pragma once
// Gemensamma typer, klasser och funktionsdeklarationer för C++-programmet.
#include <cstdint>
#include <chrono>
#include <deque>
#include <future>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <ByteTrack/BYTETracker.h>
#include <torch/script.h>
#include <yaml-cpp/yaml.h>
#include <hailo/hailort.hpp>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// En tidsmätning i millisekunder, samlad över alla anrop till samma steg.
struct TimeReading {
    size_t calls = 0;
    double total = 0, first = 0, last = 0, minimum = 0, maximum = 0;
};

// Samlar väggtid för den här körningen. Tider för hela steg innehåller även deras delsteg.
class Timings {
public:
    // Sparar tiden sedan start och lämnar tillbaka antalet millisekunder.
    double add(const std::string& name, Clock::time_point start, Clock::time_point finish = Clock::now());
    // Ger medeltiden för ett anrop till steget.
    double average(const std::string& name) const;
    // Gör tiderna till en del av programmets vanliga JSON-resultat.
    Json results() const;
    // Skriver alla mätvärden till körningens logg.
    void print() const;
private:
    std::map<std::string, TimeReading> readings;
};

// Pipelinen arbetar på huvudtråden och använder samma tidssamling i alla C++-filer.
extern Timings timings;

// De sista 300 pixlarna till höger skickas aldrig till YOLO
const int IGNORE_RIGHT_PIXELS = 300;

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

//samma föremål i klippet
struct ProductTrack {
    Json details;
    std::vector<ProductPhoto> photos;
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
    fs::path request_file, result_file, config_file, model_folder;
    std::string record_id, mode;
};

// Sökvägar och inställningar som används under ett klipp.
struct AppSettings {
    YAML::Node values;
    fs::path model_folder, vit_file, yolo_file, label_file, zones_file;
    float vit_weight = 0.7f;
};

// Kör den exporterade ViT-modellen på processorn.
class VitClassifier {
public:
    // Läser modellfilen och dess klassnamn.
    VitClassifier(const fs::path& file);
    // Ger en poäng per produktklass för en beskuren RGB-bild.
    std::vector<float> classify(const cv::Mat& image);
    std::vector<std::string> class_names;
private:
    torch::jit::Module network;
};

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
    // Äger den sidjusterade bildbufferten som Hailo läser.
    std::shared_ptr<uint8_t> input_buffer;
    // Äger modellens svarsbuffertar, en per utgång.
    std::map<std::string, std::shared_ptr<uint8_t>> output_buffers;
    std::vector<hailort::DmaMappedBuffer> mappings;
    // Äger kopplingen mellan modellens ingångar, utgångar och dessa buffertar.
    std::unique_ptr<hailort::ConfiguredInferModel::Bindings> bindings;
    std::future<HailoResult> completion;
    // Jobbet ligger sist så att det avslutas innan buffertarna frigörs.
    hailort::AsyncInferJob job;
};

// Här finns de Hailo-specifika delarna. Resten av programmet arbetar med vanliga produktrutor.
class YoloClassifier {
public:
    // Öppnar acceleratorn och läser den kompilerade YOLO-modellen.
    YoloClassifier(const AppSettings& settings);
    // Avslutar även väntande jobb innan bildbuffertarna frigörs.
    ~YoloClassifier();
    // Skickar nästa bild utan att vänta på modellkörningen.
    void send(const cv::Mat& image);
    // Väntar på den äldsta bilden och översätter dess produktrutor.
    std::vector<Detection> receive();
    std::vector<std::string> class_names;
    size_t queue_size = 1;
private:
    // Dessa smarta pekare äger Hailos resurser och frigör dem när klippet är klart.
    std::unique_ptr<hailort::VDevice> device;
    // Delat ägande av den laddade HEF-modellen.
    std::shared_ptr<hailort::InferModel> model;
    hailort::ConfiguredInferModel network;
    // Varje pekare äger en bildplats. Platserna återanvänds i turordning.
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

// Äger ByteTrack och samlar observationer och produktbilder för varje objekt.
class Tracker {
public:
    // Anpassar spårbufferten till bildfrekvensen och sparar bildstorlek och klassnamn.
    Tracker(double sample_fps, const YAML::Node& settings, cv::Size image_size, const std::vector<std::string>& class_names);
    // Kopplar produktrutor till spår och sparar deras positioner i bildrutan.
    std::vector<TrackedObject> update(const std::vector<Detection>& boxes, int frame, double time);
    std::map<size_t, ProductTrack> products;
private:
    byte_track::BYTETracker tracker;
    cv::Size image_size;
    std::vector<std::string> class_names;
};

// Hjälpfunktioner för filer, inställningar och bilder.
// Läser en JSON-fil.
Json read_json(const fs::path& file);
// Skriver JSON via en tillfällig fil så att orkestratorn får ett komplett svar.
void write_json(const fs::path& file, const Json& value);
// Ger aktuell tid i UTC.
std::string current_time();
// Läser YAML-filen och hittar modellerna och kamerornas zoner.
AppSettings load_settings(const RunOptions& options);
// Anpassar en bild till modellens storlek med utfyllda kanter.
cv::Mat resize_for_yolo(const cv::Mat& image, int width, int height, float& scale, int& left, int& top);
// Förbättrar kontrasten när det är påslaget i inställningarna.
cv::Mat prepare_image(const cv::Mat& image, bool enabled);
// Räknar ut hur stor del av två rutor som överlappar, mellan 0 och 1.
float box_overlap(const cv::Rect2f& first, const cv::Rect2f& second);
// Behåller de tydligaste rutorna och tar bort dubbletter.
std::vector<Detection> remove_duplicate_boxes(std::vector<Detection> boxes, int limit);
// Analyserar ett klipp och bygger svaret till orkestratorn.
Json process_video(const RunOptions& options, const AppSettings& settings);

// Matchar klassnamnen så att YOLO:s poäng hamnar i samma ordning som ViT:s.
std::vector<int> match_class_names(const std::vector<std::string>& yolo_names, const std::vector<std::string>& vit_names);
// Plockar ut modellens tre högsta poäng med tillhörande klassnamn.
Json top_three(std::vector<float> scores, const std::vector<std::string>& names);
// Sparar YOLO:s förslag från objektets utvalda produktbilder.
void save_yolo_predictions(ProductTrack& track, const std::vector<std::string>& class_names);
// Klassificerar objektets bilder med ViT och väger samman användbara modellförslag.
void decide_product(ProductTrack& track, VitClassifier& vit, const std::vector<int>& class_order, float vit_weight, float minimum_score);

// Väljer tydliga produktbilder från olika tidpunkter till ViT.
void keep_product_photo(ProductTrack& track, const cv::Mat& image, const cv::Rect2f& box, float score, double time, int limit, const std::vector<float>& yolo_scores);
// Returnerar tidpunkten för den tydligaste av spårets sparade produktbilder.
double best_photo_time(const ProductTrack& track);

// Läser kamerans zoner för insida och utsida.
Json read_zones(const YAML::Node& camera);
// Hittar in- och utpassager längs ett produktspår.
void find_movements(ProductTrack& track, const Json& zones, const YAML::Node& settings, double offset);
// Beräknar hur långt objektets mittpunkt har förflyttat sig i bilden.
double track_movement(const Json& observations);
