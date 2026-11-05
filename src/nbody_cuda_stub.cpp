// Używany, gdy projekt jest budowany BEZ CUDA (-DUSE_CUDA=OFF),

#include "simulation.hpp"

#include <stdexcept>

namespace nbody {

bool cuda_available() { return false; }
std::string cuda_device_name() { return "brak (zbudowano bez CUDA)"; }

void run_cuda(Bodies&, const Params&, int, CudaTimings*) {
    throw std::runtime_error("Program zbudowano bez CUDA. Użyj -DUSE_CUDA=ON.");
}

} // namespace nbody
