#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>


//-------------------
// PRE PROCESSING
//-------------------

// Gör bilden skala upp
cv::Mat resize_for_yolo(const cv::Mat& image, int width, int height, float& scale, int& left, int& top) {
    scale = std::min(float(width) / image.cols, float(height) / image.rows);
    cv::Mat resized;
    cv::resize(image, resized, {int(std::nearbyint(image.cols * scale)), int(std::nearbyint(image.rows * scale))});
    left = (width - resized.cols) / 2;
    top = (height - resized.rows) / 2;
    cv::Mat padded(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(left, top, resized.cols, resized.rows)));
    return padded; //returnerar en bild
}

// Höjer kontrasten till yolo
cv::Mat prepare_image(const cv::Mat& image, bool enabled) {
    if (!enabled) return image;
    cv::Mat lab, blurred;
    cv::cvtColor(image, lab, cv::COLOR_RGB2Lab);
    std::vector<cv::Mat> colors;
    cv::split(lab, colors);
    cv::createCLAHE(3.0, {8, 8})->apply(colors[0], colors[0]);
    cv::GaussianBlur(colors[0], blurred, {}, 1.0);
    cv::addWeighted(colors[0], 1.35, blurred, -0.35, 0, colors[0]);
    cv::merge(colors, lab);
    cv::cvtColor(lab, lab, cv::COLOR_Lab2RGB);
    return lab; //returnerar en bild
}



// Två identiska rutor ger 1. Rutor som inte möts ger 0.
float box_overlap(const cv::Rect2f& first, const cv::Rect2f& second) {
    float intersection = (first & second).area();
    float total_area = first.area() + second.area() - intersection;
    if (total_area <= 0) return 0;
    return intersection / total_area;
}

// Högst poäng får välja först. En överlappande ruta av samma klass tas bort.
std::vector<Detection> remove_duplicate_boxes(std::vector<Detection> boxes, int limit) {
    // Jämför två detektionspoäng för sorteringen.
    std::sort(boxes.begin(), boxes.end(), [](const Detection& first, const Detection& second) { return first.score > second.score; });
    std::vector<Detection> kept;
    for (const Detection& box : boxes) {
        bool duplicate = false;
        for (const Detection& other : kept) {
            if (box.label == other.label && box_overlap(box.box, other.box) > 0.7f) duplicate = true;
        }
        if (!duplicate) kept.push_back(box);
        if (int(kept.size()) >= limit) break;
    }
    return kept;
}


//--------------------
// ladda modell
//-------------------

ProductDetector::ProductDetector(const AppSettings& settings) {
    minimum_score = settings.values["detector"]["confidence"].as<float>();
    if (settings.backend == "torch") {
        cpu_model = std::make_unique<ImageModel>(settings.yolo_file, "gd-yolo26-v1");
        class_names = cpu_model->class_names;
    } else {

#ifdef GD_HAILO
        hailo_model = std::make_unique<HailoDetector>(settings);
        class_names = hailo_model->class_names;
#endif

    }
}

// Båda modellerna lämnar tillbaka samma sorts rutor till resten av programmet.
std::vector<Detection> ProductDetector::detect(const cv::Mat& image) {
    if (cpu_model) return cpu_model->detect(image, minimum_score);

#ifdef GD_HAILO
    return hailo_model->detect(image);
#endif


}

// Kör lokal YOLO och översätter rutorna tillbaka till den inskickade bildens storlek.
std::vector<Detection> ImageModel::detect(const cv::Mat& image, float minimum_score) {
    c10::InferenceMode inference;
    float scale;
    int left, top;
    cv::Mat padded = resize_for_yolo(image, image_size, image_size, scale, left, top);
    // from_blob lånar bildens pixelpekare. Flyttalsomvandlingen skapar tensorvärden för modellen.
    torch::Tensor input = torch::from_blob(padded.data, {1, image_size, image_size, 3}, torch::kUInt8);
    input = input.permute({0, 3, 1, 2}).to(torch::kFloat32) / 255.0;
    torch::IValue output = network.forward({input});
    if (output.isTuple()) output = output.toTuple()->elements()[0];
    torch::Tensor predictions = output.toTensor().contiguous();
    if (predictions.dim() != 3 || predictions.size(0) != 1) throw std::runtime_error("YOLO behöver lämna ett resultat för en bild.");
    bool has_class_scores = predictions.size(1) == int64_t(4 + class_names.size());
    if (has_class_scores) predictions = predictions.transpose(1, 2).contiguous();
    else if (predictions.size(2) != 6) throw std::runtime_error("YOLO-exportens format stöds inte.");
    auto values = predictions.accessor<float, 3>();
    std::vector<Detection> boxes;

    for (int i = 0; i < predictions.size(1); ++i) {
        auto row = values[0][i];
        Detection detection;
        detection.score = row[4];
        detection.label = 0;
        float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
        if (has_class_scores) {
            // Rå YOLO ger mittpunkt, bredd, höjd och en poäng per klass.
            detection.class_scores.resize(class_names.size());
            for (size_t label = 0; label < class_names.size(); ++label) {
                detection.class_scores[label] = row[4 + label];
                if (row[4 + label] > detection.score) { detection.score = row[4 + label]; detection.label = int(label); }
            }
            x1 = row[0] - row[2] / 2;
            y1 = row[1] - row[3] / 2;
            x2 = row[0] + row[2] / 2;
            y2 = row[1] + row[3] / 2;
        } else {
            detection.label = int(row[5]);
        }
        if (detection.score < minimum_score) continue;
        x1 = std::clamp((x1 - left) / scale, 0.0f, float(image.cols));
        y1 = std::clamp((y1 - top) / scale, 0.0f, float(image.rows));
        x2 = std::clamp((x2 - left) / scale, 0.0f, float(image.cols));
        y2 = std::clamp((y2 - top) / scale, 0.0f, float(image.rows));
        detection.box = {x1, y1, x2 - x1, y2 - y1};
        if (detection.box.area() > 0) boxes.push_back(detection);
    }
    return boxes;
}
