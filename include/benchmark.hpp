#pragma once
// Narzędzia do pomiaru czasu: rozgrzewka, powtórzenia, statystyki.

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

namespace nbody::bench {

class Timer {
public:
    void start() { t0_ = clock::now(); }
    double seconds() const {
        return std::chrono::duration<double>(clock::now() - t0_).count();
    }
private:
    using clock = std::chrono::steady_clock;
    clock::time_point t0_ = clock::now();
};

struct Stats {
    double median = 0, min = 0, max = 0, mean = 0;
};

inline Stats summarize(std::vector<double> t) {
    Stats s;
    if (t.empty()) return s;
    std::sort(t.begin(), t.end());
    const std::size_t n = t.size();
    s.median = (n % 2) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    s.min = t.front();
    s.max = t.back();
    s.mean = std::accumulate(t.begin(), t.end(), 0.0) / double(n);
    return s;
}

// Uruchamia `run` warmup + repeats razy. `setup` (np. kopia danych
// wejściowych) wykonuje się przed każdym przebiegiem i NIE jest mierzone.
// Zwraca czasy z przebiegów mierzonych (bez rozgrzewki).
template <class Setup, class Run>
std::vector<double> measure(int warmup, int repeats, Setup&& setup, Run&& run) {
    std::vector<double> times;
    times.reserve(repeats);
    for (int r = 0; r < warmup + repeats; ++r) {
        setup();
        Timer t;
        t.start();
        run();
        const double s = t.seconds();
        if (r >= warmup) times.push_back(s);
    }
    return times;
}

} // namespace nbody::bench
