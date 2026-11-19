#include "classifier.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// -------------------
// Hela bildupptagningen
//----------------------
Json process_video(const RunOptions& options, const AppSettings& settings) {
    auto stage_start = Clock::now();
    Json request = read_json(options.request_file);
    timings.add("request_read", stage_start);

    std::string camera = request.at("camera");
    if (!settings.values["cameras"][camera]) throw std::runtime_error("finns ingen kamera i conf");

    fs::path source = request.at("video_path").get<std::string>();
    if (!fs::is_regular_file(source)) throw std::runtime_error("ingen fil hittades");

    double offset = request.value("offset_seconds", 0.0);

    YAML::Node config = settings.values, tracking = config["tracking"];
    Json zones = read_zones(config["cameras"][camera]);
    if (fs::file_size(source) > config["video"]["max_upload_mb"].as<uint64_t>() * 1024 * 1024) throw std::runtime_error("Videofilen är för stor.");
    stage_start = Clock::now();
    cv::VideoCapture video(source.string());
    double fps = video.get(cv::CAP_PROP_FPS);
    int width = int(video.get(cv::CAP_PROP_FRAME_WIDTH)), height = int(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    int expected_frames = int(video.get(cv::CAP_PROP_FRAME_COUNT));
    timings.add("video_open", stage_start);
    std::cout << "Öppnat klippet: " << expected_frames << " bildrutor, " << fps << " bilder per sekund. Laddar YOLO." << std::endl;

    double requested_fps = tracking["sample_fps"].as<double>(0);

    int photo_count = std::min(5, config["classifier"]["crops_per_track"].as<int>(5)); //antal bilder vi ska ta ut (ändras i config)
    int step = 1; // Noll i config betyder att YOLO och ByteTrack använder alla bildrutor.


    if (requested_fps > 0) step = std::max(1, int(std::nearbyint(fps / requested_fps)));
    double sample_fps = fps / step;

    int visible_width = width - IGNORE_RIGHT_PIXELS; //ta bort insidan av kylskåp

    int lost_frames = std::max(1, int(std::ceil(tracking["max_gap_seconds"].as<double>() * sample_fps)));
    stage_start = Clock::now();
    HailoDetector detector(settings);
    timings.add("yolo_load", stage_start);
    stage_start = Clock::now();
    byte_track::BYTETracker tracker(30, lost_frames, 0.25f, 0.25f, 0.8f);
    std::vector<std::string> excluded = config["detector"]["excluded_labels"].as<std::vector<std::string>>(std::vector<std::string>{});
    std::map<size_t, ProductTrack> products;
    timings.add("tracking_setup", stage_start);
    std::string started = current_time();
    cv::Mat frame;
    int frame_number = 0;
    int processed_frames = 0;
    bool video_finished = false;
    // Originalets RGB-bild behålls tills rätt YOLO-svar har använts av spårningen.
    struct VideoFrame { int number; cv::Mat rgb; Clock::time_point started; };
    std::deque<VideoFrame> frames;
    auto frames_start = Clock::now();
    auto last_progress = std::chrono::steady_clock::now();
    std::cout << "YOLO använder " << detector.queue_size << " bildplatser i kön." << std::endl;


    while (!video_finished || !frames.empty()) {
        // Fyll bara de lediga platserna. En full kö väntar; den kastar aldrig bort bildrutor.
        while (!video_finished && frames.size() < detector.queue_size) {
            auto frame_start = Clock::now();
            bool has_frame = video.read(frame);
            if (!has_frame) { timings.add("video_eof", frame_start); video_finished = true; break; }
            timings.add("video_read", frame_start);
            int current_frame = frame_number++;

            if (frame_number / fps > config["video"]["max_duration_seconds"].as<double>()) throw std::runtime_error("Klippet är för långt."); //kanske ta bort??

            if (current_frame % step != 0) continue;

            stage_start = Clock::now();
            cv::Mat visible = frame(cv::Rect(0, 0, visible_width, height)).clone();
            cv::Mat rgb;
            cv::cvtColor(visible, rgb, cv::COLOR_BGR2RGB);
            timings.add("frame_prepare", stage_start);
            stage_start = Clock::now();
            cv::Mat prepared = prepare_image(rgb, config["detector"]["preprocessing"].as<bool>(false));
            timings.add("image_enhancement", stage_start);
            detector.send(prepared); //till YOLO; Hailo fortsätter medan vi arbetar vidare
            frames.push_back({current_frame, std::move(rgb), frame_start});
        }
        if (frames.empty()) break;

        // Hämta alltid äldsta bilden först, även om andra Hailo-jobb redan är klara.
        VideoFrame current = std::move(frames.front());
        frames.pop_front();
        int current_frame = current.number;
        const cv::Mat& rgb = current.rgb;
        std::vector<Detection> boxes = detector.receive();
        stage_start = Clock::now();
        boxes = remove_duplicate_boxes(boxes, config["detector"]["max_detections"].as<int>());
        timings.add("yolo_nms", stage_start);
        stage_start = Clock::now();
        std::vector<Detection> product_boxes;
        std::vector<byte_track::Object> objects;
        for (const Detection& box : boxes) {

            std::string label = detector.class_names[box.label];
            if (std::find(excluded.begin(), excluded.end(), label) != excluded.end()) continue;
            product_boxes.push_back(box);
            objects.emplace_back(byte_track::Rect<float>(box.box.x, box.box.y, box.box.width, box.box.height), box.label, box.score);
        }
        timings.add("detection_filter", stage_start);
        stage_start = Clock::now();
        auto active_tracks = tracker.update(objects);
        timings.add("tracking", stage_start);
        stage_start = Clock::now();
        std::vector<int> matches = match_boxes(active_tracks, product_boxes);
        timings.add("track_matching", stage_start);
        for (size_t i = 0; i < active_tracks.size(); ++i) {
            stage_start = Clock::now();
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
            timings.add("track_record", stage_start);
            keep_product_photo(product, rgb, box, tracked->getScore(), time, photo_count, yolo_scores);
        }
        ++processed_frames;
        timings.add("frame_latency", current.started);
        // Uppdaterar bildruteräknaren ungefär en gång per sekund medan programmet arbetar.
        auto now = std::chrono::steady_clock::now();
        if (current_frame == 0 || now - last_progress >= std::chrono::seconds(1)) {
            std::cout << "YOLO och spårning: " << current_frame + 1 << " av " << expected_frames << " bildrutor." << std::endl;
            double wall_ms = std::chrono::duration<double, std::milli>(now - frames_start).count();
            std::cout << "Tid (ms): wall_per_frame=" << wall_ms / processed_frames << ", video_read=" << timings.average("video_read") << ", frame_prepare=" << timings.average("frame_prepare") << ", yolo_prepare=" << timings.average("yolo_prepare") << ", yolo_latency=" << timings.average("yolo_latency") << ", yolo_wait=" << timings.average("yolo_wait") << ", yolo_decode=" << timings.average("yolo_decode") << ", tracking=" << timings.average("tracking") << std::endl;
            last_progress = now;
        }
    }

    double frames_ms = timings.add("frames_wall", frames_start);
    std::cout << "YOLO och spårning är klara för " << frame_number << " bildrutor." << std::endl;
    Json tracks = Json::array(), movements = Json::array(), warnings = Json::array();


    // Äger ViT-modellen och laddar den först när ett objekt behöver klassificeras.
    std::unique_ptr<VitModel> vit;
    std::vector<int> class_order;
    for (auto& item : products) {
        ProductTrack& product = item.second;
        stage_start = Clock::now();
        find_movements(product, zones, tracking, offset);
        double movement = track_movement(product.details["observations"]);
        product.details["movement"] = movement;
        timings.add("zones_movement", stage_start);
        stage_start = Clock::now();
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
        timings.add("yolo_scores", stage_start);
        if (!product.photos.empty()) {
            product.details["best_view_time"] = best_photo_time(product);
            if (!product.details["events"].empty() || movement >= tracking["minimum_movement"].as<double>()) {
                if (!vit) {
                    stage_start = Clock::now();
                    vit = std::make_unique<VitModel>(settings.vit_file);
                    class_order = match_class_names(detector.class_names, vit->class_names);
                    if (class_order.empty() && detector.class_names.size() > 1) warnings.push_back("YOLO och ViT har olika listor så vi använder vit");
                    timings.add("vit_load", stage_start);
                }

                std::cout << "ViT klassificerar objekt " << item.first << " med " << product.photos.size() << " bilder." << std::endl;
                stage_start = Clock::now();
                classify_product_vit(product, *vit, class_order, settings.vit_weight, config["classifier"]["confidence"].as<float>()); //--------------HÄR KÖR VI VIT-----------
                double milliseconds = timings.add("vit_total", stage_start);
                product.details["classification_time_ms"] = milliseconds;
                std::cout << "Tid ViT objekt " << item.first << ": " << milliseconds << " ms." << std::endl;

            }
        }
        for (const auto& event : product.details["events"]) movements.push_back({{"track_id", product.details["id"]}, {"direction", event["direction"]}, {"category", product.details["category"]}, {"classification_confidence", product.details["classification_confidence"]}, {"start", event["event_start"]}, {"end", event["event_end"]}});
        tracks.push_back(product.details);
    }

    stage_start = Clock::now();
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
    result["performance"] = {{"queue_size", detector.queue_size}, {"processed_frames", processed_frames}, {"frames_wall_ms", frames_ms}, {"average_frame_ms", processed_frames > 0 ? frames_ms / processed_frames : 0.0}, {"processed_fps", frames_ms > 0 ? processed_frames * 1000.0 / frames_ms : 0.0}};
    result["zones"] = zones;
    result["tracking_settings"] = {{"sample_fps", requested_fps}, {"stable_frames", tracking["stable_frames"].as<int>()}, {"dwell_seconds", tracking["dwell_seconds"].as<double>()}, {"max_gap_seconds", tracking["max_gap_seconds"].as<double>()}, {"minimum_movement", tracking["minimum_movement"].as<double>()}, {"max_tracks", tracking["max_tracks"].as<int>()}};
    result["backend"] = "hailo";
    result["models"] = {{"yolo", settings.yolo_file.string()}, {"vit", settings.vit_file.string()}};
    result["ignored_region"] = {{"right_pixels", IGNORE_RIGHT_PIXELS}, {"box", {double(visible_width) / width, 0.0, 1.0, 1.0}}};
    result["tracks"] = tracks;
    result["movements"] = movements;
    result["warnings"] = warnings;
    timings.add("result_build", stage_start);
    return result;
}
