#include "classifier.hpp"
#include <algorithm>
#include <cmath>

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

// Kopplar tracks till yolo och vid tveksam bedömning använder vi bara ViT
std::vector<int> match_boxes(const std::vector<byte_track::BYTETracker::STrackPtr>& tracks, const std::vector<Detection>& boxes) {
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
