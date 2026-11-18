#include "classifier.hpp"
#include <ByteTrack/BYTETracker.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

struct Crop {
    double quality, time;
    cv::Mat rgb;
};

struct Track {
    Json result;
    std::vector<Crop> crops;
};

// Zones describe what was seen on both sides of the fridge opening.
Json camera_zones(const YAML::Node& camera) {
    Json zones = {{"inside", nullptr}, {"outside", nullptr}};
    for (const std::string name : {"inside", "outside"}) {
        auto zone = camera[name];
        if (!zone || zone.IsNull()) continue;
        auto values = zone.as<std::vector<double>>();
        if (values.size() != 4 || std::any_of(values.begin(), values.end(), [](double v) { return !std::isfinite(v) || v < 0 || v > 1; }) || values[2] - values[0] < 0.01 || values[3] - values[1] < 0.01) throw std::runtime_error("Zones need normalized [left, top, right, bottom] coordinates.");
        zones[name] = values;
    }
    if (!zones["inside"].is_null() && !zones["outside"].is_null()) {
        auto a = zones["inside"].get<std::vector<double>>(), b = zones["outside"].get<std::vector<double>>();
        if (std::max(a[0], b[0]) < std::min(a[2], b[2]) && std::max(a[1], b[1]) < std::min(a[3], b[3])) throw std::runtime_error("Inside and outside zones must not overlap.");
    }
    return zones;
}

std::string locate(const Json& box, const Json& zones) {
    double x = (box[0].get<double>() + box[2].get<double>()) / 2, y = (box[1].get<double>() + box[3].get<double>()) / 2;
    for (const std::string name : {"inside", "outside"}) {
        const auto& rect = zones[name];
        if (!rect.is_null() && rect[0].get<double>() < x && x < rect[2].get<double>() && rect[1].get<double>() < y && y < rect[3].get<double>()) return name;
    }
    return "neutral";
}

void find_crossings(Json& track, const Json& zones, const YAML::Node& settings, double offset) {
    std::string stable, candidate;
    double candidate_start = 0, stable_at = 0, previous = -1;
    int count = 0;
    for (auto& sample : track["observations"]) {
        double time = sample["time"];
        if (previous >= 0 && time - previous > settings["max_gap_seconds"].as<double>()) { stable.clear(); candidate.clear(); count = 0; }
        previous = time;
        std::string zone = locate(sample["box"], zones);
        sample["zone"] = zone;
        if (zone == "neutral") { candidate.clear(); count = 0; continue; }
        if (zone != candidate) { candidate = zone; candidate_start = time; count = 1; } else ++count;
        if (count < settings["stable_frames"].as<int>() || time - candidate_start + 1e-9 < settings["dwell_seconds"].as<double>()) continue;
        if (zone != stable && !stable.empty()) track["events"].push_back({{"direction", zone == "inside" ? "in" : "out"}, {"start", stable_at}, {"end", time}, {"event_start", offset + stable_at}, {"event_end", offset + time}, {"from_zone", stable}, {"to_zone", zone}, {"source", "zone-crossing"}});
        stable = zone;
        stable_at = time;
    }
}

void keep_crop(Track& track, const cv::Mat& rgb, const cv::Rect2f& box, float score, double time, int limit) {
    int x1 = std::clamp(int(std::floor(box.x)), 0, rgb.cols), y1 = std::clamp(int(std::floor(box.y)), 0, rgb.rows);
    int x2 = std::clamp(int(std::ceil(box.x + box.width)), 0, rgb.cols), y2 = std::clamp(int(std::ceil(box.y + box.height)), 0, rgb.rows);
    if (x2 - x1 < 2 || y2 - y1 < 2) return;
    cv::Mat crop = rgb(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    double scale = std::min(1.0, 384.0 / std::max(crop.cols, crop.rows));
    if (scale < 1) cv::resize(crop, crop, {}, scale, scale, cv::INTER_AREA);
    cv::Mat grey, laplacian;
    cv::cvtColor(crop, grey, cv::COLOR_RGB2GRAY);
    cv::Laplacian(grey, laplacian, CV_64F);
    cv::Scalar mean, deviation;
    cv::meanStdDev(laplacian, mean, deviation);
    Crop view{score * std::sqrt(crop.total()) * std::log1p(deviation[0] * deviation[0]), time, crop};
    auto nearby = std::find_if(track.crops.begin(), track.crops.end(), [&](const auto& other) { return std::abs(other.time - time) < 0.5; });
    if (nearby == track.crops.end()) track.crops.push_back(view);
    else if (view.quality > nearby->quality) *nearby = view;
    std::sort(track.crops.begin(), track.crops.end(), [](const auto& a, const auto& b) { return a.quality > b.quality; });
    if (static_cast<int>(track.crops.size()) > limit) track.crops.resize(limit);
}

void classify_track(Track& track, Model& model, float confidence) {
    std::vector<float> average(model.classes.size(), 0.0f);
    track.result["classification_views"] = Json::array();
    for (const auto& crop : track.crops) {
        auto probabilities = model.classify(crop.rgb);
        for (size_t index = 0; index < average.size(); ++index) average[index] += probabilities[index] / track.crops.size();
        track.result["classification_views"].push_back(crop.time);
    }
    std::vector<size_t> order(average.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + std::min(size_t(3), order.size()), order.end(), [&](size_t a, size_t b) { return average[a] > average[b]; });
    track.result["top_categories"] = Json::array();
    for (size_t index = 0; index < std::min(size_t(3), order.size()); ++index) track.result["top_categories"].push_back({{"category", model.classes[order[index]]}, {"confidence", average[order[index]]}});
    track.result["classification_confidence"] = average[order[0]];
    track.result["category"] = average[order[0]] >= confidence ? model.classes[order[0]] : "unknown";
    track.result["label_source"] = "grocery-vit";
}

Json run_clip(const Options& options, const Settings& settings) {
    const auto request = read_json(options.request);
    if (request.at("schema_version") != 2 || request.at("capture_id") != options.record_id) throw std::runtime_error("Request schema or capture ID does not match.");
    for (const std::string key : {"event_id", "captured_at", "camera", "video_path"}) if (!request.at(key).is_string() || request.at(key).get<std::string>().empty()) throw std::runtime_error("Request needs " + key);
    fs::path source = request.at("video_path").get<std::string>();
    if (!source.is_absolute() || !fs::is_regular_file(source)) throw std::runtime_error("video_path must be an existing absolute path.");
    std::string camera = request.at("camera");
    if (!settings.config["cameras"][camera]) throw std::runtime_error("Unknown camera: " + camera);
    const auto zones = camera_zones(settings.config["cameras"][camera]);
    double offset = request.value("offset_seconds", 0.0);
    if (!std::isfinite(offset) || offset < 0) throw std::runtime_error("offset_seconds must be finite and nonnegative.");
    const auto config = settings.config, tracking = config["tracking"], detector_settings = config["detector"];
    if (fs::file_size(source) > config["video"]["max_upload_mb"].as<uint64_t>() * 1024 * 1024) throw std::runtime_error("Video exceeds the configured size limit.");
    if (!fs::is_regular_file(settings.vit)) throw std::runtime_error("Copy an exported vit.pth into " + settings.vit.parent_path().string());
    cv::VideoCapture video(source.string());
    double fps = video.get(cv::CAP_PROP_FPS);
    int frames = int(video.get(cv::CAP_PROP_FRAME_COUNT)), width = int(video.get(cv::CAP_PROP_FRAME_WIDTH)), height = int(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (!video.isOpened() || !std::isfinite(fps) || fps <= 0 || frames < 1 || width < 2 || height < 2) throw std::runtime_error("Video cannot be decoded or has invalid timing.");
    if (frames / fps > config["video"]["max_duration_seconds"].as<double>() || int64_t(width) * height > config["video"]["max_pixels"].as<int64_t>()) throw std::runtime_error("Video exceeds the configured duration or dimensions.");
    int stride = std::max(1, int(std::nearbyint(fps / tracking["sample_fps"].as<double>())));
    double sample_fps = fps / stride;
    auto detector = make_detector(settings);
    auto excluded_values = detector_settings["excluded_labels"].as<std::vector<std::string>>(std::vector<std::string>{});
    std::set<std::string> excluded(excluded_values.begin(), excluded_values.end());
    int track_buffer = std::max(1, int(std::ceil(tracking["max_gap_seconds"].as<double>() * sample_fps)));
    auto bytetrack = tracking["bytetrack"] ? tracking["bytetrack"] : YAML::Node(YAML::NodeType::Map);
    // This ByteTrack port scales the buffer by frame_rate/30, so pass 30 and a sampled-frame count.
    byte_track::BYTETracker tracker(30, track_buffer, bytetrack["track_high_thresh"].as<float>(0.25f), bytetrack["new_track_thresh"].as<float>(0.25f), bytetrack["match_thresh"].as<float>(0.8f));
    fs::create_directories(options.output.parent_path());
    std::map<size_t, Track> tracks;
    const std::string started = now();
    cv::Mat frame;
    int frame_index = 0;

    // Read the orchestrator's clip directly. No capture, segmentation or conversion here.
    while (video.read(frame)) {
        int current = frame_index++;
        if (current == 0 && !cv::imwrite((options.output.parent_path() / "poster.jpg").string(), frame)) throw std::runtime_error("Cannot save poster.jpg.");
        if (current % stride) continue;
        double time = current / fps;
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        auto rows = detector->detect(prepare_detector(rgb, detector_settings["preprocessing"].as<bool>(false)));
        rows = suppress_duplicates(std::move(rows), detector_settings["max_detections"].as<int>());
        std::vector<byte_track::Object> objects;
        for (const auto& row : rows) {
            if (row.label < 0 || static_cast<size_t>(row.label) >= detector->classes.size()) throw std::runtime_error("Detector class ID is outside its labels.");
            if (!excluded.count(detector->classes[row.label])) objects.emplace_back(byte_track::Rect<float>(row.box.x, row.box.y, row.box.width, row.box.height), row.label, row.score);
        }
        for (const auto& object : tracker.update(objects)) {
            const auto rect = object->getRect();
            cv::Rect2f box(rect.x(), rect.y(), rect.width(), rect.height());
            std::vector<double> normalized{std::clamp(double(box.x / width), 0.0, 1.0), std::clamp(double(box.y / height), 0.0, 1.0), std::clamp(double((box.x + box.width) / width), 0.0, 1.0), std::clamp(double((box.y + box.height) / height), 0.0, 1.0)};
            if (normalized[0] >= normalized[2] || normalized[1] >= normalized[3]) continue;
            size_t id = object->getTrackId();
            if (!tracks.count(id)) {
                if (tracks.size() >= tracking["max_tracks"].as<size_t>()) throw std::runtime_error("Too many tracks; use a shorter clip.");
                std::string label = "item";
                float best_iou = 0;
                for (const auto& row : rows) {
                    float intersection = (row.box & box).area(), iou = intersection / std::max(1e-6f, row.box.area() + box.area() - intersection);
                    if (iou > best_iou) { best_iou = iou; label = detector->classes[row.label]; }
                }
                tracks[id].result = {{"id", std::to_string(id)}, {"category", "unknown"}, {"classification_confidence", nullptr}, {"label_source", "unclassified"}, {"source_label", label}, {"observations", Json::array()}, {"events", Json::array()}, {"ignored", false}};
            }
            auto& track = tracks[id];
            track.result["observations"].push_back({{"frame", current}, {"time", time}, {"box", normalized}, {"detection_confidence", object->getScore()}});
            keep_crop(track, rgb, box, object->getScore(), time, config["classifier"]["crops_per_track"].as<int>());
        }
    }
    if (frame_index < std::max(1, frames - 2)) throw std::runtime_error("Video decoding stopped early; no partial result was accepted.");

    // Confirm crossings first, then classify a few clear views of moving products.
    Json results = Json::array(), movements = Json::array();
    std::unique_ptr<Model> classifier;
    for (auto& entry : tracks) {
        auto& track = entry.second;
        find_crossings(track.result, zones, tracking, offset);
        double min_x = 1, max_x = 0, min_y = 1, max_y = 0;
        for (const auto& observation : track.result["observations"]) {
            auto box = observation["box"].get<std::vector<double>>();
            double x = (box[0] + box[2]) / 2, y = (box[1] + box[3]) / 2;
            min_x = std::min(min_x, x); max_x = std::max(max_x, x); min_y = std::min(min_y, y); max_y = std::max(max_y, y);
        }
        double displacement = std::hypot(max_x - min_x, max_y - min_y);
        track.result["movement"] = displacement;
        if (!track.crops.empty()) {
            std::string crop_file = "track-" + std::to_string(entry.first) + ".jpg";
            cv::Mat bgr;
            cv::cvtColor(track.crops[0].rgb, bgr, cv::COLOR_RGB2BGR);
            if (!cv::imwrite((options.output.parent_path() / crop_file).string(), bgr)) throw std::runtime_error("Cannot save track crop.");
            track.result["crop_file"] = crop_file;
            if (!track.result["events"].empty() || displacement >= tracking["minimum_movement"].as<double>()) {
                if (!classifier) classifier = std::make_unique<Model>(settings.vit, "gd-vit-rgb-v1");
                classify_track(track, *classifier, config["classifier"]["confidence"].as<float>());
            }
        }
        for (const auto& event : track.result["events"]) movements.push_back({{"track_id", track.result["id"]}, {"direction", event["direction"]}, {"category", track.result["category"]}, {"classification_confidence", track.result["classification_confidence"]}, {"start", event["event_start"]}, {"end", event["event_end"]}});
        results.push_back(track.result);
    }
    std::sort(movements.begin(), movements.end(), [](const auto& a, const auto& b) { return a.at("start").template get<double>() < b.at("start").template get<double>(); });
    Json warnings = Json::array();
    if (zones["inside"].is_null() || zones["outside"].is_null()) warnings.push_back("Set both zones before interpreting in/out movements.");
    Json tracking_settings = {{"sample_fps", tracking["sample_fps"].as<double>()}, {"stable_frames", tracking["stable_frames"].as<int>()}, {"dwell_seconds", tracking["dwell_seconds"].as<double>()}, {"max_gap_seconds", tracking["max_gap_seconds"].as<double>()}, {"minimum_movement", tracking["minimum_movement"].as<double>()}, {"max_tracks", tracking["max_tracks"].as<int>()}};
    return {{"schema_version", 2}, {"capture_id", options.record_id}, {"event_id", request["event_id"]}, {"mode", options.mode}, {"captured_at", request["captured_at"]}, {"camera", camera}, {"offset_seconds", offset}, {"started_at", started}, {"completed_at", now()}, {"source", source.string()}, {"video", {{"fps", fps}, {"frames", frames}, {"width", width}, {"height", height}, {"duration", frames / fps}}}, {"sample_fps", sample_fps}, {"zones", zones}, {"tracking_settings", tracking_settings}, {"backend", settings.backend}, {"models", {{"yolo", settings.yolo.string()}, {"vit", settings.vit.string()}}}, {"sensor_metadata", request.value("sensor_metadata", Json::object())}, {"attachments", request.value("attachments", Json::array())}, {"tracks", results}, {"movements", movements}, {"warnings", warnings}};
}
