#include "simulation.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace nbody;

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void compare(const Bodies& expected, const Bodies& actual, double tolerance) {
    require(expected.size() == actual.size(), "Niepoprawny rozmiar wyniku");
    const std::vector<real>* a[] = {&expected.x, &expected.y, &expected.z,
        &expected.vx, &expected.vy, &expected.vz, &expected.ax, &expected.ay, &expected.az};
    const std::vector<real>* b[] = {&actual.x, &actual.y, &actual.z,
        &actual.vx, &actual.vy, &actual.vz, &actual.ax, &actual.ay, &actual.az};
    for (int field = 0; field < 9; ++field)
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const double x = (*a[field])[i], y = (*b[field])[i];
            require(std::isfinite(x) && std::isfinite(y), "NaN/Inf w stanie");
            require(std::abs(x - y) <= tolerance * std::max(1.0, std::abs(x)),
                    "Roznica polozen, predkosci lub przyspieszen");
        }
    require(expected.m == actual.m, "Masy zostaly zmienione");
}

template <class F> void rejects(F run) {
    bool rejected = false;
    try { run(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Brak odrzucenia niepoprawnego wejscia");
}
}

int main() {
    try {
        Params p;
        const bool gpu = cuda_available();
        const double cpu_tol = sizeof(real) == 4 ? 1e-5 : 1e-12;
        const double gpu_tol = sizeof(real) == 4 ? 3e-4 : 1e-10;
        // Granice kafli oraz puste i jednoelementowe uklady. Kontrola calego
        // stanu wykrywa tez bledy ostatniego kick, niewidoczne w polozeniach.
        for (std::size_t n : {0, 1, 2, 31, 255, 256, 257, 513}) {
            Bodies initial = make_random(n, 42);
            for (std::size_t i = 0; i < n; ++i)
                initial.m[i] *= real(0.5) + real(i % 7) / real(7);
            for (int steps : {0, 1, 7}) {
                Bodies ref = initial;
                run_seq(ref, p, steps);
                for (int threads : {1, 2, 4}) {
                    Bodies b = initial;
                    run_omp(b, p, steps, threads);
                    compare(ref, b, cpu_tol);
                }
                if (gpu) for (auto kernel : {CudaKernel::naive, CudaKernel::tiled}) {
                    Bodies b = initial;
                    CudaTimings timing;
                    run_cuda(b, p, steps, &timing, kernel);
                    compare(ref, b, gpu_tol);
                    require(timing.h2d_s >= 0 && timing.kernel_s >= 0 && timing.d2h_s >= 0,
                            "Ujemny czas CUDA");
                }
            }
        }
        // Jeden swobodny obiekt: ruch jednostajny, brak sily wlasnej.
        Bodies lone(1);
        lone.m[0] = real(1); lone.vx[0] = real(2);
        run_seq(lone, p, 10);
        require(std::abs(double(lone.x[0]) - 20 * double(p.dt)) < cpu_tol,
                "Niepoprawny ruch jednostajny");

        Bodies bad(2);
        bad.x.pop_back();
        rejects([&] { run_seq(bad, p, 1); });
        rejects([&] { run_omp(bad, p, 1, 2); });
        if (gpu) rejects([&] { run_cuda(bad, p, 1); });
        Bodies b = make_two_body();
        rejects([&] { run_seq(b, p, -1); });
        rejects([&] { run_omp(b, p, 1, -1); });
        Params invalid = p; invalid.eps = 0;
        rejects([&] { run_seq(b, invalid, 1); });
        invalid = p; invalid.dt = std::numeric_limits<real>::quiet_NaN();
        rejects([&] { run_seq(b, invalid, 1); });
        rejects([&] { max_position_diff(b, Bodies(1)); });
        Bodies nan = b; nan.x[0] = std::numeric_limits<real>::quiet_NaN();
        require(std::isinf(max_position_diff(b, nan)), "NaN ukryty przez porownanie");
        std::cout << "Stan CPU i przypadki brzegowe OK; CUDA: "
                  << (gpu ? "oba kernele OK" : "pominieta") << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
