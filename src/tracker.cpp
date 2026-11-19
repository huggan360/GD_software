
// Dokumentation: https://github.com/FoundationVision/ByteTrack
// C++-biblioteket vi använder: https://github.com/Vertical-Beach/ByteTrack-cpp
#include "program.hpp"
#include <algorithm>
#include <cmath>

//-------------------
// Starta ByteTrack
//-------------------

// ByteTrack: Zhang et al. (2022), Multi-Object Tracking by Associating Every Detection Box.
// https://doi.org/10.1007/978-3-031-20047-2_1

// Behåll försvunna objekt i max_gap_seconds sekunder.
Tracker::Tracker(double fps, const YAML::Node& settings, cv::Size image_size) : tracker(30, std::max(1, int(std::ceil(settings["max_gap_seconds"].as<double>() * fps))), 0.25f, 0.25f, 0.8f), image_size(image_size) {}

//---------------------------------
// Följ produkterna i nästa bildruta
//---------------------------------

// ByteTrack ger produkterna samma ID medan de rör sig.
std::vector<TrackedObject> Tracker::update(const std::vector<Detection>& boxes, double time) {
    auto start = Clock::now();
    std::vector<byte_track::Object> objects;
    for (const Detection& box : boxes) objects.emplace_back(byte_track::Rect<float>(box.box.x, box.box.y, box.box.width, box.box.height), box.label, box.score);
    auto active_tracks = tracker.update(objects);
    timings.add("tracking", start);
    std::vector<TrackedObject> result;
    std::vector<bool> used(boxes.size(), false);

    for (const auto& tracked : active_tracks) {
        start = Clock::now();
        const auto& rectangle = tracked->getRect();
        cv::Rect2f box(rectangle.x(), rectangle.y(), rectangle.width(), rectangle.height());
        box &= cv::Rect2f(0, 0, float(image_size.width - IGNORE_RIGHT_PIXELS), float(image_size.height));
        if (box.area() <= 0) continue;
        int id = int(tracked->getTrackId());
        if (!products.count(id)) products[id].details = {{"id", std::to_string(id)}, {"category", "unknown"}, {"classification_confidence", nullptr}};

        cv::Rect2f position(box.x / image_size.width, box.y / image_size.height, box.width / image_size.width, box.height / image_size.height);
        products[id].observations.push_back({time, position});
        timings.add("track_record", start);
        start = Clock::now();

        int match = -1;
        float best_overlap = 0.3f;
        for (int i = 0; i < int(boxes.size()); ++i) {
            float overlap = box_overlap(box, boxes[i].box);
            if (!used[i] && overlap > best_overlap) { best_overlap = overlap; match = i; }
        }
        std::vector<float> yolo_scores;
        if (match >= 0) { yolo_scores = boxes[match].class_scores; used[match] = true; }
        timings.add("track_matching", start);
        result.push_back({id, box, tracked->getScore(), std::move(yolo_scores)});
    }
    return result;
}

//-------------------
// Bestäm in eller ut efter hela spåret
//-------------------

// metod från Calle et al. (2022), Automatic Counting Passenger System using Online Visual Appearance Multi-Object Tracking
// https://doi.org/10.25046/aj070514
// Vi jämför positionen längs x axeln med tidigare positioners medelvärde
// Vi räknar gränszonen som inne. Vi sparar bara nettobyte mellan spårets början och slut.
void find_movements(ProductTrack& track, int image_width, double offset) {
    double line = double(image_width - IGNORE_RIGHT_PIXELS - BORDER_ZONE_PIXELS) / image_width;
    const Observation& first = track.observations.front();
    bool started_inside = first.box.x + first.box.width / 2 >= line;
    bool inside = started_inside;
    double sum = 0;
    int points = 0;
    for (const Observation& observation : track.observations) {
        double x = observation.box.x + observation.box.width / 2;
        bool current_inside = x >= line;
        if (points > 0 && current_inside != (sum / points >= line)) {
            inside = current_inside;
            sum = 0;
            points = 0;
        }
        sum += x;
        ++points;
    }
    track.details["inside"] = inside;
    if (inside != started_inside) track.events.push_back({{"direction", inside ? "in" : "out"}, {"start", offset + first.time}, {"end", offset + track.observations.back().time}});
}

// Mäter rörelsen så att även produkter utan en passage kan skickas till ViT.
double track_movement(const std::vector<Observation>& observations) {
    double left = 1, right = 0, top = 1, bottom = 0;
    for (const Observation& observation : observations) {
        double x = observation.box.x + observation.box.width / 2, y = observation.box.y + observation.box.height / 2;
        left = std::min(left, x); right = std::max(right, x);
        top = std::min(top, y); bottom = std::max(bottom, y);
    }
    return std::hypot(right - left, bottom - top);
}
