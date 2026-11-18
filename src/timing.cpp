#include "classifier.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

Timings timings;

// Räknar millisekunder och uppdaterar stegets summa, medelvärde och yttervärden.
double Timings::add(const std::string& name, Clock::time_point start, Clock::time_point finish) {
    double milliseconds = std::chrono::duration<double, std::milli>(finish - start).count();
    TimeReading& reading = readings[name];
    if (reading.calls == 0) reading.first = reading.minimum = reading.maximum = milliseconds;
    ++reading.calls;
    reading.total += milliseconds;
    reading.last = milliseconds;
    reading.minimum = std::min(reading.minimum, milliseconds);
    reading.maximum = std::max(reading.maximum, milliseconds);
    return milliseconds;
}

// Noll betyder att steget ännu inte har körts.
double Timings::average(const std::string& name) const {
    auto found = readings.find(name);
    if (found == readings.end()) return 0;
    return found->second.total / found->second.calls;
}

// Sparar även första och sista tiden så att en långsam första ViT-bild syns tydligt.
Json Timings::results() const {
    Json result = Json::object();
    for (const auto& [name, reading] : readings) result[name] = {{"calls", reading.calls}, {"total_ms", reading.total}, {"average_ms", reading.total / reading.calls}, {"minimum_ms", reading.minimum}, {"maximum_ms", reading.maximum}, {"first_ms", reading.first}, {"last_ms", reading.last}};
    return result;
}

// Alla tider är väggtid. När CPU och Hailo arbetar samtidigt överlappar deras tider.
void Timings::print() const {
    std::cout << "Tidsmätning i ms: överlappande steg kan inte summeras. Latens innehåller köväntan; frames_wall är bildrutornas verkliga sammanlagda tid.\n" << std::fixed << std::setprecision(3);
    for (const auto& [name, reading] : readings) std::cout << name << ": calls=" << reading.calls << ", total_ms=" << reading.total << ", average_ms=" << reading.total / reading.calls << ", minimum_ms=" << reading.minimum << ", maximum_ms=" << reading.maximum << ", first_ms=" << reading.first << ", last_ms=" << reading.last << '\n';
    std::cout.flush();
}
