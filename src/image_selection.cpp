#include "program.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

//----------------------------------------
// Välj produktbilder till ViT
// Normalt tar vi ut 5 bildrutor från hela tracket och då vill vi ha bilder från lite olika ställen men som också har bra kvalite och det är detta vi försöker åstadkomma här
// bilden får max vara 512px för att vit inte ska gå för långsamt (men kan undersöka detta vidare såklart)
//----------------------------------------


bool keep_product_photo(ProductTrack& track, const cv::Mat& image, const cv::Rect2f& box, double time) {
    auto stage_start = Clock::now();

    int left = std::clamp(int(std::floor(box.x)), 0, image.cols); //floor och ceil avrundar olika
    int top = std::clamp(int(std::floor(box.y)), 0, image.rows);
    int right = std::clamp(int(std::ceil(box.x + box.width)), 0, image.cols);
    int bottom = std::clamp(int(std::ceil(box.y + box.height)), 0, image.rows);

    // om den är mindre än 32x32 pixlar så är det kört
    if (right - left < 32 || bottom - top < 32) {
        timings.add("photo_quality", stage_start);
        return false;
    }

    cv::Mat product_area = image(cv::Rect(left, top, right - left, bottom - top));
    double scale = std::min(32.0, 512.0 / std::max(product_area.cols, product_area.rows));

    //följande beräknar bildkvaliten
    cv::Mat grey, edges, photo;
    cv::cvtColor(product_area, grey, cv::COLOR_RGB2GRAY);
    if (scale < 1) cv::resize(grey, grey, {}, scale, scale, cv::INTER_AREA);

    // Skärpan uppskattas först med Variance of Laplacian
    // Referens: Ali & Mahmood (2018), Applied Sciences, 8(5), 807.
    // https://doi.org/10.3390/app8050807
    cv::Laplacian(grey, edges, CV_64F);
    cv::Scalar average, deviation;
    cv::meanStdDev(edges, average, deviation);
    double quality = deviation[0] * deviation[0];

    timings.add("photo_quality", stage_start);

    stage_start = Clock::now();

    // Om vi redan sparat 5 så kollar vi om den har bättre kvalitet
    if (track.photos.size() >= 5) {
        int worst = 0;
        for (int i = 1; i < int(track.photos.size()); ++i) {
            if (track.photos[i].quality < track.photos[worst].quality) worst = i;
        }

        // Behåll den gamla bilden om den har lika bra eller bättre kvalitet.
        if (quality <= track.photos[worst].quality) {
            timings.add("photo_selection", stage_start);
            return false;
        }
        track.photos.erase(track.photos.begin() + worst);
    }
    timings.add("photo_selection", stage_start);

    stage_start = Clock::now();

    // Skapa produktbilden som senare används av ViT.
    if (scale < 1) cv::resize(product_area, photo, {}, scale, scale, cv::INTER_AREA);
    else photo = product_area.clone();
    track.photos.push_back({quality, time, photo, {}});

    timings.add("photo_save", stage_start);
    return true;
}

// Väljer produktbilden mitt i tracket för att visa mest för att undvika att ta den bästa för detta kräver ytterligare 2 sökloopar
Json product_result(const ProductTrack& track) {
    int middle = track.observations.size() / 2;
    const auto& observation = track.observations[middle];
    const auto& box = observation.box;

    Json result = track.details;
    result["start"] = track.observations.front().time;
    result["end"] = track.observations.back().time;
    result["time"] = observation.time;
    result["box"] = {box.x,box.y,box.x + box.width,box.y + box.height};

    return result;
}
