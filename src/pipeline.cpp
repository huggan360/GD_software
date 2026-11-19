#include "program.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Kör ett färdigt videoklipp genom modellerna, spårningen och produktbeslutet.
Json process_video(const RunOptions& options, const AppSettings& settings) {
    //-------------------
    // Steg A: Läs begäran och kamerans inställningar
    //-------------------
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
    //-------------------
    // Steg B: Öppna videon och bestäm vilka bildrutor som ska analyseras
    //-------------------
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

    //-------------------
    // Steg C: Ladda YOLO och starta spårningen
    //-------------------
    stage_start = Clock::now();
    YoloClassifier yolo(settings);
    timings.add("yolo_load", stage_start);
    stage_start = Clock::now();
    Tracker tracker(sample_fps, tracking, cv::Size(width, height), yolo.class_names);
    auto& products = tracker.products;
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
    std::cout << "YOLO använder " << yolo.queue_size << " bildplatser i kön." << std::endl;


    while (!video_finished || !frames.empty()) {
        //-------------------
        // Steg D: Läs, förbered och skicka nästa bilder till YOLO
        //-------------------
        // Fyll bara de lediga platserna. En full kö väntar; den kastar aldrig bort bildrutor.
        while (!video_finished && frames.size() < yolo.queue_size) {
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
            yolo.send(prepared); //till YOLO; Hailo fortsätter medan vi arbetar vidare
            frames.push_back({current_frame, std::move(rgb), frame_start});
        }
        if (frames.empty()) break;

        //-------------------
        // Steg E: Hämta YOLO-svaret och uppdatera spårningen
        //-------------------
        // Hämta alltid äldsta bilden först, även om andra Hailo-jobb redan är klara.
        VideoFrame current = std::move(frames.front());
        frames.pop_front();
        int current_frame = current.number;
        const cv::Mat& rgb = current.rgb;
        std::vector<Detection> boxes = yolo.receive();
        double time = current_frame / fps;
        std::vector<TrackedObject> objects = tracker.update(boxes, current_frame, time);

        //-------------------
        // Steg F: Behåll tydliga produktbilder från olika tidpunkter till ViT
        //-------------------
        for (const TrackedObject& object : objects) keep_product_photo(products[object.id], rgb, object.box, object.score, time, photo_count, object.yolo_scores);

        //-------------------
        // Steg G: Visa framsteg och tidsmätningar
        //-------------------
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


    //-------------------
    // Steg H: Bedöm objektens rörelse och passager mellan zonerna
    //-------------------
    // Äger ViT-modellen och laddar den först när ett objekt behöver klassificeras.
    std::unique_ptr<VitClassifier> vit;
    std::vector<int> class_order;
    for (auto& item : products) {
        ProductTrack& product = item.second;
        stage_start = Clock::now();
        find_movements(product, zones, tracking, offset);
        double movement = track_movement(product.details["observations"]);
        product.details["movement"] = movement;
        timings.add("zones_movement", stage_start);
        save_yolo_predictions(product, yolo.class_names);
        if (!product.photos.empty()) {
            product.details["best_view_time"] = best_photo_time(product);
            if (!product.details["events"].empty() || movement >= tracking["minimum_movement"].as<double>()) {
                //-------------------
                // Steg I: Klassificera produktbilderna och bestäm produktnamnet
                //-------------------
                if (!vit) {
                    stage_start = Clock::now();
                    vit = std::make_unique<VitClassifier>(settings.vit_file);
                    class_order = match_class_names(yolo.class_names, vit->class_names);
                    if (class_order.empty() && yolo.class_names.size() > 1) warnings.push_back("YOLO och ViT har olika listor så vi använder vit");
                    timings.add("vit_load", stage_start);
                }

                std::cout << "ViT klassificerar objekt " << item.first << " med " << product.photos.size() << " bilder." << std::endl;
                stage_start = Clock::now();
                decide_product(product, *vit, class_order, settings.vit_weight, config["classifier"]["confidence"].as<float>()); // Här vägs modellernas förslag samman.
                double milliseconds = timings.add("vit_total", stage_start);
                product.details["classification_time_ms"] = milliseconds;
                std::cout << "Tid ViT objekt " << item.first << ": " << milliseconds << " ms." << std::endl;

            }
        }
        for (const auto& event : product.details["events"]) movements.push_back({{"track_id", product.details["id"]}, {"direction", event["direction"]}, {"category", product.details["category"]}, {"classification_confidence", product.details["classification_confidence"]}, {"start", event["event_start"]}, {"end", event["event_end"]}});
        tracks.push_back(product.details);
    }

    //-------------------
    // Steg J: Bygg svaret till orkestratorn
    //-------------------
    stage_start = Clock::now();
    // Sorterar passagerna efter tidpunkten i klippet.
    std::sort(movements.begin(), movements.end(), [](const Json& first, const Json& second) { return first["start"].get<double>() < second["start"].get<double>(); });

    if (zones["inside"].is_null() || zones["outside"].is_null()) warnings.push_back("inga zoner");

    Json result = request;
    result["mode"] = options.mode;
    result["started_at"] = started;
    result["completed_at"] = current_time();
    result["source"] = source.string();
    result["video"] = {{"fps", fps}, {"frames", frame_number}, {"width", width}, {"height", height}, {"duration", frame_number / fps}};
    result["sample_fps"] = sample_fps;
    result["performance"] = {{"queue_size", yolo.queue_size}, {"processed_frames", processed_frames}, {"frames_wall_ms", frames_ms}, {"average_frame_ms", processed_frames > 0 ? frames_ms / processed_frames : 0.0}, {"processed_fps", frames_ms > 0 ? processed_frames * 1000.0 / frames_ms : 0.0}};
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
