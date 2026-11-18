#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// -------------------
// Hela bildupptagningen
//----------------------
Json process_video(const RunOptions& options, const AppSettings& settings) {
    Json request = read_json(options.request_file);

    std::string camera = request.at("camera");
    if (!settings.values["cameras"][camera]) throw std::runtime_error("finns ingen kamera i conf");

    fs::path source = request.at("video_path").get<std::string>();
    if (!fs::is_regular_file(source)) throw std::runtime_error("ingen fil hittades");

    double offset = request.value("offset_seconds", 0.0);

    YAML::Node config = settings.values, tracking = config["tracking"];
    Json zones = read_zones(config["cameras"][camera]);
    if (fs::file_size(source) > config["video"]["max_upload_mb"].as<uint64_t>() * 1024 * 1024) throw std::runtime_error("Videofilen är för stor.");
    cv::VideoCapture video(source.string());
    double fps = video.get(cv::CAP_PROP_FPS);
    int width = int(video.get(cv::CAP_PROP_FRAME_WIDTH)), height = int(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    int expected_frames = int(video.get(cv::CAP_PROP_FRAME_COUNT));

    double requested_fps = tracking["sample_fps"].as<double>(0);

    int photo_count = std::min(5, config["classifier"]["crops_per_track"].as<int>(5)); //antal bilder vi ska ta ut (ändras i config)
    int step = 1; // Noll i config betyder att YOLO och ByteTrack använder alla bildrutor.


    if (requested_fps > 0) step = std::max(1, int(std::nearbyint(fps / requested_fps)));
    double sample_fps = fps / step;

    int visible_width = width - IGNORE_RIGHT_PIXELS; //ta bort insidan av kylskåp

    int lost_frames = std::max(1, int(std::ceil(tracking["max_gap_seconds"].as<double>() * sample_fps)));
    ProductDetector detector(settings);
    byte_track::BYTETracker tracker(30, lost_frames, 0.25f, 0.25f, 0.8f);
    std::vector<std::string> excluded = config["detector"]["excluded_labels"].as<std::vector<std::string>>(std::vector<std::string>{});
    std::map<size_t, ProductTrack> products;
    std::string started = current_time();
    cv::Mat frame;
    int frame_number = 0;


    while (video.read(frame)) {
        int current_frame = frame_number++;

        if (frame_number / fps > config["video"]["max_duration_seconds"].as<double>()) throw std::runtime_error("Klippet är för långt."); //kanske ta bort??

        if (current_frame % step != 0) continue;

        cv::Mat visible = frame(cv::Rect(0, 0, visible_width, height)).clone();
        cv::Mat rgb;
        cv::cvtColor(visible, rgb, cv::COLOR_BGR2RGB);
        std::vector<Detection> boxes = detector.detect(prepare_image(rgb, config["detector"]["preprocessing"].as<bool>(false))); //till YOLO
        boxes = remove_duplicate_boxes(boxes, config["detector"]["max_detections"].as<int>());
        std::vector<Detection> product_boxes;
        std::vector<byte_track::Object> objects;
        for (const Detection& box : boxes) {

            std::string label = detector.class_names[box.label];
            if (std::find(excluded.begin(), excluded.end(), label) != excluded.end()) continue;
            product_boxes.push_back(box);
            objects.emplace_back(byte_track::Rect<float>(box.box.x, box.box.y, box.box.width, box.box.height), box.label, box.score);
        }
        auto active_tracks = tracker.update(objects);
        std::vector<int> matches = match_boxes(active_tracks, product_boxes);
        for (size_t i = 0; i < active_tracks.size(); ++i) {
            // Lånar ByteTracks smarta pekare. Trackern äger fortfarande spåret.
            const auto& tracked = active_tracks[i];
            const auto& rectangle = tracked->getRect();
            cv::Rect2f box(rectangle.x(), rectangle.y(), rectangle.width(), rectangle.height());
            box = box & cv::Rect2f(0, 0, float(visible_width), float(height));
            if (box.area() <= 0) continue;
            size_t id = tracked->getTrackId();
            if (!products.count(id)) {
                std::string label = "item";
                if (matches[i] >= 0) label = detector.class_names[product_boxes[matches[i]].label];
                products[id].details = {{"id", std::to_string(id)}, {"category", "unknown"}, {"classification_confidence", nullptr}, {"label_source", "unclassified"}, {"source_label", label}, {"observations", Json::array()}, {"events", Json::array()}, {"ignored", false}};
            }
            ProductTrack& product = products[id];
            double time = current_frame / fps;
            // Koordinaterna är fortfarande relativa till HELA bilden, inklusive den bortvalda högersidan.
            std::vector<double> position{double(box.x / width), double(box.y / height), double((box.x + box.width) / width), double((box.y + box.height) / height)};
            product.details["observations"].push_back({{"frame", current_frame}, {"time", time}, {"box", position}, {"detection_confidence", tracked->getScore()}});
            std::vector<float> yolo_scores;
            if (matches[i] >= 0) yolo_scores = product_boxes[matches[i]].class_scores;
            keep_product_photo(product, rgb, box, tracked->getScore(), time, photo_count, yolo_scores);
        }
    }

    Json tracks = Json::array(), movements = Json::array(), warnings = Json::array();


    std::unique_ptr<ImageModel> vit;
    std::vector<int> class_order;
    for (auto& item : products) {
        ProductTrack& product = item.second;
        find_movements(product, zones, tracking, offset);
        double movement = track_movement(product.details["observations"]);
        product.details["movement"] = movement;
        // Sparar YOLO:s egna förslag från de valda bilderna, även om ViT har andra klassnamn.
        std::vector<float> yolo_scores(detector.class_names.size(), 0);
        int yolo_photos = 0;
        for (const ProductPhoto& photo : product.photos) {
            if (photo.yolo_scores.size() != yolo_scores.size()) continue;
            for (size_t i = 0; i < yolo_scores.size(); ++i) yolo_scores[i] += photo.yolo_scores[i];
            ++yolo_photos;
        }
        product.details["yolo_top_categories"] = Json::array();
        if (yolo_photos > 0) {
            for (float& score : yolo_scores) score /= yolo_photos;
            product.details["yolo_top_categories"] = top_three(yolo_scores, detector.class_names);
        }
        if (!product.photos.empty()) {
            product.details["best_view_time"] = best_photo_time(product);
            if (!product.details["events"].empty() || movement >= tracking["minimum_movement"].as<double>()) {
                if (!vit) {
                    vit = std::make_unique<ImageModel>(settings.vit_file, "gd-vit-rgb-v1");
                    class_order = match_class_names(detector.class_names, vit->class_names);
                    if (class_order.empty() && detector.class_names.size() > 1) warnings.push_back("YOLO och ViT har olika listor så vi använder vit");
                }

                classify_product_vit(product, *vit, class_order, settings.vit_weight, config["classifier"]["confidence"].as<float>()); //--------------HÄR KÖR VI VIT-----------

            }
        }
        for (const auto& event : product.details["events"]) movements.push_back({{"track_id", product.details["id"]}, {"direction", event["direction"]}, {"category", product.details["category"]}, {"classification_confidence", product.details["classification_confidence"]}, {"start", event["event_start"]}, {"end", event["event_end"]}});
        tracks.push_back(product.details);
    }

    // sortera först in osv
    std::sort(movements.begin(), movements.end(), [](const Json& first, const Json& second) { return first["start"].get<double>() < second["start"].get<double>(); });

    if (zones["inside"].is_null() || zones["outside"].is_null()) warnings.push_back("inga zoner");

    Json result = request;
    result["mode"] = options.mode;
    result["started_at"] = started;
    result["completed_at"] = current_time();
    result["source"] = source.string();
    result["video"] = {{"fps", fps}, {"frames", frame_number}, {"width", width}, {"height", height}, {"duration", frame_number / fps}};
    result["sample_fps"] = sample_fps;
    result["zones"] = zones;
    result["tracking_settings"] = {{"sample_fps", requested_fps}, {"stable_frames", tracking["stable_frames"].as<int>()}, {"dwell_seconds", tracking["dwell_seconds"].as<double>()}, {"max_gap_seconds", tracking["max_gap_seconds"].as<double>()}, {"minimum_movement", tracking["minimum_movement"].as<double>()}, {"max_tracks", tracking["max_tracks"].as<int>()}};
    result["backend"] = settings.backend;
    result["models"] = {{"yolo", settings.yolo_file.string()}, {"vit", settings.vit_file.string()}};
    result["ignored_region"] = {{"right_pixels", IGNORE_RIGHT_PIXELS}, {"box", {double(visible_width) / width, 0.0, 1.0, 1.0}}};
    result["tracks"] = tracks;
    result["movements"] = movements;
    result["warnings"] = warnings;
    return result;
}
