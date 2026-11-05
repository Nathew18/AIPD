// Wspólne funkcje: warunki początkowe i kontrola poprawności.
#include "simulation.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace nbody {

namespace {

// Liczba z [0, 1) liczona ręcznie z surowego wyjścia mt19937_64.
// (std::uniform_real_distribution może dawać różne wyniki na różnych
//  kompilatorach - tak mamy identyczne dane na Windows/macOS/Linux.)
double uniform01(std::mt19937_64& rng) {
    return double(rng() >> 11) * (1.0 / 9007199254740992.0); // / 2^53
}

// Losowy punkt w kuli jednostkowej (losowanie z odrzucaniem).
void random_in_ball(std::mt19937_64& rng, double& x, double& y, double& z) {
    do {
        x = 2.0 * uniform01(rng) - 1.0;
        y = 2.0 * uniform01(rng) - 1.0;
        z = 2.0 * uniform01(rng) - 1.0;
    } while (x * x + y * y + z * z > 1.0);
}

} // namespace

Bodies make_random(std::size_t n, std::uint64_t seed) {
    Bodies b(n);
    std::mt19937_64 rng(seed);

    std::vector<double> px(n), py(n), pz(n), vx(n), vy(n), vz(n);
    const double mass = 1.0 / double(n);
    double cx = 0, cy = 0, cz = 0, cvx = 0, cvy = 0, cvz = 0;

    for (std::size_t i = 0; i < n; ++i) {
        random_in_ball(rng, px[i], py[i], pz[i]);
        // Prędkości z kuli o promieniu 1 => układ w przybliżeniu w równowadze
        // wirialnej dla jednorodnej kuli o masie 1 i promieniu 1 (przy G = 1).
        random_in_ball(rng, vx[i], vy[i], vz[i]);
        cx += px[i]; cy += py[i]; cz += pz[i];
        cvx += vx[i]; cvy += vy[i]; cvz += vz[i];
    }
    cx /= double(n); cy /= double(n); cz /= double(n);
    cvx /= double(n); cvy /= double(n); cvz /= double(n);

    // Przejście do układu środka masy: pęd całkowity = 0.
    for (std::size_t i = 0; i < n; ++i) {
        b.x[i] = real(px[i] - cx);
        b.y[i] = real(py[i] - cy);
        b.z[i] = real(pz[i] - cz);
        b.vx[i] = real(vx[i] - cvx);
        b.vy[i] = real(vy[i] - cvy);
        b.vz[i] = real(vz[i] - cvz);
        b.m[i] = real(mass);
    }
    return b;
}

Bodies make_two_body() {
    // m1 = m2 = 0.5, odległość d = 1, każde ciało krąży po okręgu r = 0.5.
    // G m^2 / d^2 = m v^2 / r  =>  v = 0.5,  okres T = 2*pi*r / v = 2*pi.
    Bodies b(2);
    b.m[0] = b.m[1] = real(0.5);
    b.x[0] = real(-0.5); b.x[1] = real(0.5);
    b.vy[0] = real(-0.5); b.vy[1] = real(0.5);
    return b;
}

double total_energy(const Bodies& b, const Params& p) {
    // Liczone w double niezależnie od `real`, żeby kontrola była wiarygodna.
    const std::size_t n = b.size();
    const double eps2 = double(p.eps) * double(p.eps);
    double kin = 0, pot = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v2 = double(b.vx[i]) * b.vx[i] + double(b.vy[i]) * b.vy[i] +
                          double(b.vz[i]) * b.vz[i];
        kin += 0.5 * double(b.m[i]) * v2;
        for (std::size_t j = i + 1; j < n; ++j) {
            const double dx = double(b.x[j]) - b.x[i];
            const double dy = double(b.y[j]) - b.y[i];
            const double dz = double(b.z[j]) - b.z[i];
            // Potencjał zgodny z siłą "softened": -G m_i m_j / sqrt(r^2 + eps^2)
            pot -= double(p.G) * double(b.m[i]) * double(b.m[j]) /
                   std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
    }
    return kin + pot;
}

double momentum_magnitude(const Bodies& b) {
    double px = 0, py = 0, pz = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        px += double(b.m[i]) * b.vx[i];
        py += double(b.m[i]) * b.vy[i];
        pz += double(b.m[i]) * b.vz[i];
    }
    return std::sqrt(px * px + py * py + pz * pz);
}

double max_position_diff(const Bodies& a, const Bodies& b) {
    double d = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = double(a.x[i]) - b.x[i];
        const double dy = double(a.y[i]) - b.y[i];
        const double dz = double(a.z[i]) - b.z[i];
        d = std::max(d, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return d;
}

} // namespace nbody
