#pragma once
// Model danych: stan wszystkich ciał w układzie SoA (Structure of Arrays).
// Osobne tablice dla każdej współrzędnej => dostęp ciągły w pamięci,
// co pomaga wektoryzacji na CPU i coalescingowi na GPU.

#include <cstddef>
#include <vector>

namespace nbody {

#ifdef NBODY_USE_FLOAT
using real = float;
#else
using real = double;
#endif

// Parametry fizyczne i numeryczne symulacji (jednostki umowne, G = 1).
struct Params {
    real G   = real(1.0);    // stała grawitacji
    real dt  = real(0.001);  // krok czasowy
    real eps = real(0.01);   // softening: musi być > 0 (zapobiega dzieleniu przez 0)
};

struct Bodies {
    std::vector<real> x, y, z;     // położenie
    std::vector<real> vx, vy, vz;  // prędkość
    std::vector<real> ax, ay, az;  // przyspieszenie (bufor roboczy integratora)
    std::vector<real> m;           // masa

    Bodies() = default;
    explicit Bodies(std::size_t n) { resize(n); }

    std::size_t size() const { return m.size(); }

    void resize(std::size_t n) {
        for (auto* v : {&x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &m})
            v->assign(n, real(0));
    }
};

} // namespace nbody
