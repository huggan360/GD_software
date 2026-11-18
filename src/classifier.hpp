#pragma once
#include <cstdint>
#include <chrono>
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
#ifdef GD_HAILO
#include <hailo/hailort.hpp>
#endif

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
    double add(const std::string& name, Clock::time_point start);
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

// Det som orkestratorn skickar när programmet startas.
struct RunOptions {
    fs::path request_file, result_file, config_file, model_folder;
    std::string record_id, mode;
};

// Sökvägar och inställningar som används under ett klipp.
struct AppSettings {
    YAML::Node values;
    fs::path model_folder, vit_file, yolo_file, label_file, zones_file;
    std::string backend;
    float vit_weight = 0.7f;
};

// Kör exporterade PyTorch-modeller på processorn. Samma laddare används för ViT och lokal YOLO.
class ImageModel {
public:
    // Läser modellfilen och dess klassnamn.
    ImageModel(const fs::path& file, const std::string& expected_format);
    // Ger en poäng per produktklass för en beskuren RGB-bild.
    std::vector<float> classify(const cv::Mat& image);
    // Hittar produktrutor med CPU-versionen av YOLO.
    std::vector<Detection> detect(const cv::Mat& image, float minimum_score);
    std::vector<std::string> class_names;
    int image_size = 640;
private:
    torch::jit::Module network;
};

#ifdef GD_HAILO
// Här finns de Hailo-specifika delarna. Resten av programmet arbetar med vanliga produktrutor.
class HailoDetector {
public:
    // Öppnar acceleratorn och läser den kompilerade YOLO-modellen.
    HailoDetector(const AppSettings& settings);
    // Skickar en RGB-bild till Hailo och läser tillbaka produktrutorna.
    std::vector<Detection> detect(const cv::Mat& image);
    std::vector<std::string> class_names;
private:
    hailort::Hef model;
    // Dessa smarta pekare äger Hailos resurser och frigör dem när klippet är klart.
    std::unique_ptr<hailort::VDevice> device;
    // Hailo delar nätverksgruppen mellan sina interna delar.
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network;
    // Äger aktiveringen som håller nätverket igång.
    std::unique_ptr<hailort::ActivatedNetworkGroup> activation;
    // Äger in- och utflödena för bilddata.
    std::unique_ptr<hailort::InferVStreams> streams;
    std::vector<hailo_vstream_info_t> outputs;
    std::map<std::string, std::vector<float>> output_values;
    std::vector<std::string> class_outputs;
    std::string input_name;
    int width = 640, height = 640;
    float minimum_score = 0.15f;
};
#endif

// Väljer Hailo eller CPU en gång när klippet öppnas. Ingen arvshierarki behövs.
class ProductDetector {
public:
    // Laddar den detektor som valts i video.yaml.
    ProductDetector(const AppSettings& settings);
    // Hittar objekt i den del av bilden som ska analyseras.
    std::vector<Detection> detect(const cv::Mat& image);
    std::vector<std::string> class_names;
private:
    float minimum_score;
    // Äger CPU-modellen när backend är torch; annars är pekaren tom.
    std::unique_ptr<ImageModel> cpu_model;
#ifdef GD_HAILO
    // Äger Hailo-detektorn när backend är hailo.
    std::unique_ptr<HailoDetector> hailo_model;
#endif
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
// Klassificerar spårets utvalda bilder med ViT och kombinerar användbara YOLO-poäng.
void classify_product_vit(ProductTrack& track, ImageModel& vit, const std::vector<int>& class_order, float vit_weight, float minimum_score);

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
// Kopplar ByteTracks spår till rätt YOLO-rutor i samma bildruta.
std::vector<int> match_boxes(const std::vector<byte_track::BYTETracker::STrackPtr>& tracks, const std::vector<Detection>& boxes);
