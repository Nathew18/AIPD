// Program do uruchamiania symulacji, testów poprawności i benchmarków.
//
// Przykłady:
//   nbody --mode test
//   nbody --mode all --n 1024,2048,4096 --steps 10 --repeats 5 --check
//   nbody --mode omp --n 8192 --threads 1,2,4,8 --csv wyniki.csv
#include "benchmark.hpp"
#include "simulation.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nbody;

namespace {

struct Options {
    std::string mode = "all";            // seq | omp | cuda | all | test
    std::vector<long> n = {1024};
    std::vector<long> threads;           // puste = maks. dostępne
    int steps = 10;
    int repeats = 5;
    int warmup = 1;
    std::uint64_t seed = 42;
    bool check = false;                  // energia + różnica względem seq
    std::string csv;                     // puste = stdout
    Params params;
};

void print_help() {
    std::cout <<
        "Uzycie: nbody [opcje]\n"
        "  --mode seq|omp|cuda|all|test  wariant (domyslnie all)\n"
        "  --n 1024[,2048,...]           liczba cial (mozna podac liste)\n"
        "  --steps 10                    liczba krokow symulacji\n"
        "  --threads 1,2,4,8             watki OpenMP (domyslnie maks.)\n"
        "  --repeats 5                   liczba mierzonych powtorzen\n"
        "  --warmup 1                    przebiegi rozgrzewkowe (niemierzone)\n"
        "  --seed 42                     ziarno warunkow poczatkowych\n"
        "  --dt 0.001  --eps 0.01        krok czasowy, softening\n"
        "  --check                       dryf energii i roznica wzgledem seq\n"
        "  --csv plik.csv                dopisz wyniki do pliku zamiast na ekran\n";
}

std::vector<long> parse_list(const std::string& s) {
    std::vector<long> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) out.push_back(std::stol(item));
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Brak wartosci dla " + a);
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { print_help(); std::exit(0); }
        else if (a == "--mode")    o.mode = next();
        else if (a == "--n")       o.n = parse_list(next());
        else if (a == "--threads") o.threads = parse_list(next());
        else if (a == "--steps")   o.steps = std::stoi(next());
        else if (a == "--repeats") o.repeats = std::stoi(next());
        else if (a == "--warmup")  o.warmup = std::stoi(next());
        else if (a == "--seed")    o.seed = std::stoull(next());
        else if (a == "--dt")      o.params.dt = real(std::stod(next()));
        else if (a == "--eps")     o.params.eps = real(std::stod(next()));
        else if (a == "--check")   o.check = true;
        else if (a == "--csv")     o.csv = next();
        else throw std::runtime_error("Nieznana opcja: " + a + " (zobacz --help)");
    }
    if (o.params.eps <= 0) throw std::runtime_error("--eps musi byc > 0");
    return o;
}

const char* real_name() { return sizeof(real) == 4 ? "float" : "double"; }

// ============================================================ testy poprawności
struct TestRunner {
    int failed = 0;
    void check(const std::string& name, double value, double limit) {
        const bool ok = std::isfinite(value) && value <= limit;
        std::printf("  [%s] %-46s %.3e (limit %.1e)\n", ok ? " OK " : "FAIL",
                    name.c_str(), value, limit);
        if (!ok) ++failed;
    }
};

int run_tests() {
    const bool f = sizeof(real) == 4;
    TestRunner t;
    std::printf("Testy poprawnosci (typ: %s)\n", real_name());

    // 1. Dwa ciała na orbicie kołowej: po jednym okresie wracają na start.
    {
        std::printf("\n1) Orbita dwoch cial (jeden pelny okres)\n");
        Params p;
        p.eps = real(1e-6);                     // praktycznie bez softeningu
        const int steps = 10000;
        p.dt = real(2.0 * 3.14159265358979323846 / steps);
        Bodies start = make_two_body();
        Bodies b = start;
        const double e0 = total_energy(b, p);
        run_seq(b, p, steps);
        t.check("blad polozenia po okresie", max_position_diff(start, b), f ? 5e-3 : 1e-4);
        t.check("wzgledny dryf energii", std::abs((total_energy(b, p) - e0) / e0), f ? 1e-4 : 1e-7);
    }

    // 2. Losowa chmura: zachowanie energii i pędu.
    Params p;
    const int steps = 100;
    const Bodies start = make_random(512, 123);
    Bodies ref = start;
    {
        std::printf("\n2) Zasady zachowania (N = 512, %d krokow, seq)\n", steps);
        const double e0 = total_energy(ref, p);
        run_seq(ref, p, steps);
        t.check("wzgledny dryf energii", std::abs((total_energy(ref, p) - e0) / e0), f ? 1e-3 : 1e-4);
        t.check("|ped calkowity|", momentum_magnitude(ref), f ? 1e-5 : 1e-12);
    }

    // 3. OpenMP daje (prawie) te same wyniki co wersja sekwencyjna.
    {
        std::printf("\n3) OpenMP vs sekwencyjna (%d watkow)\n", omp_max_threads());
        Bodies b = start;
        run_omp(b, p, steps, 0);
        t.check("maks. roznica polozen", max_position_diff(ref, b), f ? 1e-4 : 1e-10);
    }

    // 4. CUDA vs sekwencyjna (jeśli dostępna).
    if (cuda_available()) {
        std::printf("\n4) CUDA vs sekwencyjna (%s)\n", cuda_device_name().c_str());
        Bodies b = start;
        run_cuda(b, p, steps);
        t.check("maks. roznica polozen", max_position_diff(ref, b), f ? 1e-3 : 1e-8);
    } else {
        std::printf("\n4) CUDA: pominieto (%s)\n", cuda_device_name().c_str());
    }

    std::printf("\n%s (%d nieudanych)\n", t.failed ? "TESTY NIEUDANE" : "WSZYSTKIE TESTY OK", t.failed);
    return t.failed ? 1 : 0;
}

// ============================================================ benchmark
void run_benchmark(const Options& o) {
    std::vector<std::string> modes;
    if (o.mode == "all") {
        modes = {"seq", "omp"};
        if (cuda_available()) modes.push_back("cuda");
        else std::cerr << "[info] CUDA niedostepna - pomijam wariant cuda\n";
    } else if (o.mode == "seq" || o.mode == "omp" || o.mode == "cuda") {
        modes = {o.mode};
    } else {
        throw std::runtime_error("Nieznany tryb: " + o.mode);
    }

    std::ofstream file;
    bool need_header = true;
    if (!o.csv.empty()) {
        std::ifstream existing(o.csv);
        need_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
        file.open(o.csv, std::ios::app);
        if (!file) throw std::runtime_error("Nie moge otworzyc " + o.csv);
    }
    std::ostream& out = o.csv.empty() ? std::cout : file;
    if (need_header)
        out << "mode,real,n,steps,threads,repeats,median_s,min_s,max_s,mean_s,"
               "ms_per_step,ginteractions_per_s,energy_rel_drift,max_diff_vs_seq,"
               "cuda_h2d_s,cuda_kernel_s,cuda_d2h_s\n";

    std::cerr << "[info] typ: " << real_name() << ", watki OpenMP (maks.): "
              << omp_max_threads() << ", GPU: " << cuda_device_name() << "\n";

    for (long n : o.n) {
        const Bodies initial = make_random(std::size_t(n), o.seed);
        const double e0 = o.check ? total_energy(initial, o.params) : 0.0;

        Bodies reference;  // wynik seq do porównań (liczony raz, jeśli --check)
        if (o.check) {
            reference = initial;
            run_seq(reference, o.params, o.steps);
        }

        for (const std::string& mode : modes) {
            std::vector<long> thread_list = {1};
            if (mode == "omp")
                thread_list = o.threads.empty() ? std::vector<long>{omp_max_threads()} : o.threads;

            for (long th : thread_list) {
                std::cerr << "[run] " << mode << " n=" << n
                          << (mode == "omp" ? " threads=" + std::to_string(th) : "") << "\n";
                Bodies b;
                CudaTimings ct, ct_sum;  // suma tylko z przebiegow mierzonych
                int run_no = 0;
                auto times = bench::measure(
                    o.warmup, o.repeats, [&] { b = initial; },
                    [&] {
                        if (mode == "seq") run_seq(b, o.params, o.steps);
                        else if (mode == "omp") run_omp(b, o.params, o.steps, int(th));
                        else {
                            run_cuda(b, o.params, o.steps, &ct);
                            if (run_no++ < o.warmup) return;
                            ct_sum.h2d_s += ct.h2d_s;
                            ct_sum.kernel_s += ct.kernel_s;
                            ct_sum.d2h_s += ct.d2h_s;
                        }
                    });
                const bench::Stats s = bench::summarize(times);
                const double pairs = double(n) * double(n) * o.steps;

                out << mode << ',' << real_name() << ',' << n << ',' << o.steps << ','
                    << th << ',' << o.repeats << ',' << s.median << ',' << s.min << ','
                    << s.max << ',' << s.mean << ',' << 1000.0 * s.median / o.steps << ','
                    << pairs / s.median / 1e9 << ',';
                if (o.check) {
                    out << std::abs((total_energy(b, o.params) - e0) / e0) << ','
                        << (mode == "seq" ? 0.0 : max_position_diff(reference, b));
                } else {
                    out << ',';
                }
                out << ',';
                if (mode == "cuda") {
                    const double runs = double(o.repeats);
                    out << ct_sum.h2d_s / runs << ',' << ct_sum.kernel_s / runs << ','
                        << ct_sum.d2h_s / runs;
                } else {
                    out << ",,";
                }
                out << '\n' << std::flush;
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        if (o.mode == "test") return run_tests();
        run_benchmark(o);
    } catch (const std::exception& e) {
        std::cerr << "Blad: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
