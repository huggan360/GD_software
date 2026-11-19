#include "program.hpp"
#include <iomanip>
#include <iostream>

Timings timings;

// Räknar millisekunder
double Timings::add(const std::string& name, Clock::time_point start, Clock::time_point finish) {
    double milliseconds = std::chrono::duration<double, std::milli>(finish - start).count();

    TimeReading& reading = readings[name];
    ++reading.calls;

    reading.total += milliseconds;
    return milliseconds;
}

// sammanställer
void Timings::print() const {
    std::cout << "\nTider i ms. Överlappande steg kan inte summeras.\n" << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(24) << "Steg" << std::right << std::setw(10) << "Anrop" << std::setw(16) << "Totalt ms" << std::setw(16) << "Medel ms" << '\n';
    for (const auto& [name, reading] : readings) {
        std::cout << std::left << std::setw(24) << name << std::right << std::setw(10) << reading.calls << std::setw(16) << reading.total << std::setw(16) << reading.total / reading.calls << '\n';
    }
}

