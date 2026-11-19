#include "program.hpp"
#include <algorithm>
#include <cmath>

//-------------------
// Starta ByteTrack
//-------------------

// Bufferten anges i analyserade bildrutor, så samma max_gap_seconds gäller vid olika bildfrekvenser.
Tracker::Tracker(double sample_fps, const YAML::Node& settings, cv::Size image_size, const std::vector<std::string>& class_names) : tracker(30, std::max(1, int(std::ceil(settings["max_gap_seconds"].as<double>() * sample_fps))), 0.25f, 0.25f, 0.8f), image_size(image_size), class_names(class_names) {}

// Kopplar tracks till yolo och vid tveksam bedömning använder vi bara ViT
static std::vector<int> match_boxes(const std::vector<byte_track::BYTETracker::STrackPtr>& tracks, const std::vector<Detection>& boxes) {
    std::vector<int> matches(tracks.size(), -1), used(boxes.size(), 0);
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& rectangle = tracks[i]->getRect();
        cv::Rect2f track_box(rectangle.x(), rectangle.y(), rectangle.width(), rectangle.height());
        float best_overlap = 0, second_overlap = 0;
        int best_box = -1;
        for (size_t j = 0; j < boxes.size(); ++j) {
            float overlap = box_overlap(track_box, boxes[j].box);
            if (overlap > best_overlap) { second_overlap = best_overlap; best_overlap = overlap; best_box = int(j); }
            else second_overlap = std::max(second_overlap, overlap);
        }
        if (best_box >= 0 && best_overlap >= 0.3f && best_overlap - second_overlap >= 0.05f) {
            matches[i] = best_box;
            ++used[best_box];
        }
    }
    // En ruta får bara ge poäng till ett spår.
    for (int& match : matches) if (match >= 0 && used[match] != 1) match = -1;
    return matches;
}

//-------------------
// Följ objekten i nästa bildruta
//-------------------

// All kontakt med ByteTrack finns här. Pipelinen får tillbaka vanliga objekt och produktrutor.
std::vector<TrackedObject> Tracker::update(const std::vector<Detection>& boxes, int frame, double time) {
    auto start = Clock::now();
    std::vector<byte_track::Object> objects;
    for (const Detection& box : boxes) objects.emplace_back(byte_track::Rect<float>(box.box.x, box.box.y, box.box.width, box.box.height), box.label, box.score);
    auto active_tracks = tracker.update(objects);
    timings.add("tracking", start);
    start = Clock::now();
    std::vector<int> matches = match_boxes(active_tracks, boxes);
    timings.add("track_matching", start);
    std::vector<TrackedObject> result;
    for (size_t i = 0; i < active_tracks.size(); ++i) {
        start = Clock::now();
        // Lånar ByteTracks smarta pekare. Trackern äger fortfarande spåret.
        const auto& tracked = active_tracks[i];
        const auto& rectangle = tracked->getRect();
        cv::Rect2f box(rectangle.x(), rectangle.y(), rectangle.width(), rectangle.height());
        box = box & cv::Rect2f(0, 0, float(image_size.width - IGNORE_RIGHT_PIXELS), float(image_size.height));
        if (box.area() <= 0) continue;
        size_t id = tracked->getTrackId();
        if (!products.count(id)) {
            std::string label = "item";
            if (matches[i] >= 0) label = class_names[boxes[matches[i]].label];
            products[id].details = {{"id", std::to_string(id)}, {"category", "unknown"}, {"classification_confidence", nullptr}, {"label_source", "unclassified"}, {"source_label", label}, {"observations", Json::array()}, {"events", Json::array()}, {"ignored", false}};
        }
        // Positionerna räknas mot hela originalbilden, även den bortvalda högersidan.
        std::vector<double> position{double(box.x / image_size.width), double(box.y / image_size.height), double((box.x + box.width) / image_size.width), double((box.y + box.height) / image_size.height)};
        products[id].details["observations"].push_back({{"frame", frame}, {"time", time}, {"box", position}, {"detection_confidence", tracked->getScore()}});
        std::vector<float> yolo_scores;
        if (matches[i] >= 0) yolo_scores = boxes[matches[i]].class_scores;
        result.push_back({id, box, tracked->getScore(), std::move(yolo_scores)});
        timings.add("track_record", start);
    }
    return result;
}

//-------------------
// Läs zonerna och bedöm in och ut
//-------------------

// Läser in zonerna
Json read_zones(const YAML::Node& camera) {
    Json zones = {{"inside", nullptr}, {"outside", nullptr}};
    for (const std::string name : {"inside", "outside"}) {
        if (!camera[name] || camera[name].IsNull()) continue;
        std::vector<double> rectangle = camera[name].as<std::vector<double>>();
        zones[name] = rectangle;
    }
    if (!zones["inside"].is_null() && !zones["outside"].is_null()) {
        std::vector<double> inside = zones["inside"], outside = zones["outside"];
    }
    return zones;
}

//---------------------
// Tracking in/ut
//-------------------

// mitt på objektet bedömmer hur den flyttas inne, ute eller neutral
std::string find_zone(const Json& box, const Json& zones) {
    double x = (box[0].get<double>() + box[2].get<double>()) / 2;
    double y = (box[1].get<double>() + box[3].get<double>()) / 2;
    for (const std::string name : {"inside", "outside"}) {
        if (zones[name].is_null()) continue;
        std::vector<double> rectangle = zones[name];
        if (x > rectangle[0] && x < rectangle[2] && y > rectangle[1] && y < rectangle[3]) return name;
    }
    return "inget";
}

// in och ut räknas när objektet syns på båda sidor (kanske ska byta denna till något mer robust?)
void find_movements(ProductTrack& track, const Json& zones, const YAML::Node& settings, double offset) {
    std::string previous_zone, candidate_zone;
    double candidate_start = 0, last_stable_time = 0, previous_time = -1;
    int observations_in_zone = 0;
    for (auto& observation : track.details["observations"]) {
        double time = observation["time"];
        if (previous_time >= 0 && time - previous_time > settings["max_gap_seconds"].as<double>()) {
            previous_zone.clear();
            candidate_zone.clear();
            observations_in_zone = 0;
        }
        previous_time = time;
        std::string zone = find_zone(observation["box"], zones);
        observation["zone"] = zone;
        if (zone == "inget") {
            candidate_zone.clear();
            observations_in_zone = 0;
            continue;
        }
        if (zone != candidate_zone) {
            candidate_zone = zone;
            candidate_start = time;
            observations_in_zone = 0;
        }
        ++observations_in_zone;
        if (observations_in_zone < settings["stable_frames"].as<int>()) continue;
        if (time - candidate_start + 1e-9 < settings["dwell_seconds"].as<double>()) continue;
        if (!previous_zone.empty() && zone != previous_zone) {
            std::string direction = "out";
            if (zone == "inside") direction = "in";
            Json event = {{"direction", direction}, {"start", last_stable_time}, {"end", time}, {"event_start", offset + last_stable_time}, {"event_end", offset + time}, {"from_zone", previous_zone}, {"to_zone", zone}, {"source", "zone-crossing"}};
            track.details["events"].push_back(event);
        }
        previous_zone = zone;
        last_stable_time = time;
    }
}

// Här klassifierar vi utan en in/ut om vi har en riktig förflyttning
double track_movement(const Json& observations) {
    double left = 1, right = 0, top = 1, bottom = 0;
    for (const auto& observation : observations) {
        std::vector<double> box = observation["box"];
        double x = (box[0] + box[2]) / 2, y = (box[1] + box[3]) / 2;
        left = std::min(left, x); right = std::max(right, x);
        top = std::min(top, y); bottom = std::max(bottom, y);
    }
    return std::hypot(right - left, bottom - top);
}
