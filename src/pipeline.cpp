#include "program.hpp"
#include <algorithm>
#include <deque>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Kör ett färdigt videoklipp genom modellerna, spårningen och produktbeslutet.
Json process_video(const RunOptions& options, const Settings& settings) {

//-------------------
// Steg A: Läs begäran och kamerans inställningar
//-------------------


    auto stage_start = Clock::now();
    Json request = read_json(options.request_file);
    timings.add("request_read", stage_start);

    std::string camera = request.at("camera");
    if (!settings.values["cameras"][camera]) throw std::runtime_error("finns ingen kamera i conf");

    std::filesystem::path source = request.at("video_path").get<std::string>();

    double offset = request.value("offset_seconds", 0.0);

    YAML::Node config = settings.values, tracking = config["tracking"];
    Json zones = read_zones(config["cameras"][camera]);


//-------------------
// Steg B: Öppna videon och bestäm vilka bildrutor som ska analyseras
//-------------------

    stage_start = Clock::now();
    cv::VideoCapture video(source.string());
    double fps = video.get(cv::CAP_PROP_FPS);
    int width = int(video.get(cv::CAP_PROP_FRAME_WIDTH)), height = int(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (fps <= 0 || width <= IGNORE_RIGHT_PIXELS || height <= 0) throw std::runtime_error("Kan inte läsa videon.");
    timings.add("video_open", stage_start);

    int photo_count = std::clamp(config["classifier"]["crops_per_track"].as<int>(5), 1, 5); //antal bilder vi ska ta ut (ändras i config)

    int visible_width = width - IGNORE_RIGHT_PIXELS; //ta bort insidan av kylskåp

    //-------------------
    // Steg C: Ladda YOLO och starta spårningen
    //-------------------

    stage_start = Clock::now();
    YoloClassifier yolo(settings);
    timings.add("yolo_load", stage_start);
    stage_start = Clock::now();
    Tracker tracker(fps, tracking, cv::Size(width, height));
    auto& products = tracker.products;
    timings.add("tracking_setup", stage_start);
    cv::Mat frame;
    int frame_number = 0;
    bool video_finished = false;

    // Originalets RGB-bild behålls tills rätt YOLO-svar har använts av spårningen.
    struct VideoFrame { double time; cv::Mat rgb; Clock::time_point started; };

    std::deque<VideoFrame> frames;
    auto frames_start = Clock::now();


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
            stage_start = Clock::now();
            cv::Mat rgb;
            cv::cvtColor(frame(cv::Rect(0, 0, visible_width, height)), rgb, cv::COLOR_BGR2RGB);
            timings.add("frame_prepare", stage_start);
            yolo.send(rgb); //till YOLO; Hailo fortsätter medan vi arbetar vidare
            frames.push_back({frame_number++ / fps, std::move(rgb), frame_start});
        }
        if (frames.empty()) break;

//-------------------
// Steg E: Hämta YOLO-svaret och uppdatera spårningen
//-------------------

        // Hämta alltid äldsta bilden först, även om andra Hailo-jobb redan är klara.
        VideoFrame current = std::move(frames.front());
        frames.pop_front();
        const cv::Mat& rgb = current.rgb;
        std::vector<Detection> boxes = yolo.receive();
        double time = current.time;
        std::vector<TrackedObject> objects = tracker.update(boxes, time);

//-------------------
// Steg F: Behåll tydliga produktbilder från olika tidpunkter till ViT
//-------------------

        for (const TrackedObject& object : objects) keep_product_photo(products[object.id], rgb, object.box, object.score, time, photo_count, object.yolo_scores);

        timings.add("frame_latency", current.started);
    }

    timings.add("frames_wall", frames_start);
    Json tracks = Json::array(), movements = Json::array();


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
        double movement = track_movement(product.observations);
        timings.add("zones_movement", stage_start);
        save_yolo_predictions(product, yolo.class_names);
        if (!product.photos.empty() && (!product.events.empty() || movement >= tracking["minimum_movement"].as<double>())) {


                //-------------------
                // Steg I: Klassificera produktbilderna och bestäm produktnamnet
                //-------------------


                if (!vit) {
                    stage_start = Clock::now();
                    vit = std::make_unique<VitClassifier>(settings.vit_file);
                    class_order = match_class_names(yolo.class_names, vit->class_names);
                    timings.add("vit_load", stage_start);
                }

                stage_start = Clock::now();
                decide_product(product, *vit, class_order, settings.vit_weight, config["classifier"]["confidence"].as<float>()); // Här vägs modellernas förslag samman.
                timings.add("vit_total", stage_start);

        }
        for (const auto& event : product.events) movements.push_back({{"track_id", product.details["id"]}, {"direction", event["direction"]}, {"category", product.details["category"]}, {"classification_confidence", product.details["classification_confidence"]}, {"start", event["start"]}, {"end", event["end"]}});
        tracks.push_back(product_result(product));
    }

    //-------------------
    // Steg J: bygg json till orkestratorn
    //-------------------

    stage_start = Clock::now();
    // Sorterar passagerna efter tidpunkten i klippet.
    std::sort(movements.begin(), movements.end(), [](const Json& first, const Json& second) { return first["start"].get<double>() < second["start"].get<double>(); });

    Json result = {{"schema_version", 2}, {"capture_id", request.at("capture_id")}, {"event_id", request.at("event_id")}, {"mode", options.mode}, {"tracks", tracks}, {"movements", movements}};
    timings.add("result_build", stage_start);

    return result;
}
