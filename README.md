# Symulacja grawitacyjnego problemu N-ciał

Porównanie wydajności trzech implementacji tego samego algorytmu
(bezpośrednia metoda O(N²) + integrator leapfrog):

| Wariant | Plik | Status |
|---|---|---|
| CPU sekwencyjnie | `src/nbody_seq.cpp` | ✅ gotowe, przetestowane |
| CPU + OpenMP | `src/nbody_openmp.cpp` | ✅ gotowe, przetestowane |
| GPU + CUDA (wersja podstawowa) | `src/nbody_cuda.cu` | ⚠️ napisane, **nieprzetestowane** (wymaga karty NVIDIA) |
| GPU + CUDA (tiling `__shared__`) | — | ⏳ do zrobienia |

## Struktura

```
include/
  body.hpp          dane ciał (SoA), parametry fizyczne, typ `real` (float/double)
  simulation.hpp    wspólny interfejs + funkcja sił accel_on_body (__host__ __device__)
  benchmark.hpp     pomiar czasu: rozgrzewka, powtórzenia, mediana
src/
  main.cpp          argumenty, testy poprawności, benchmark -> CSV
  simulation.cpp    warunki początkowe, energia, pęd, porównywanie wyników
  nbody_seq.cpp     wariant sekwencyjny (wzorzec)
  nbody_openmp.cpp  wariant OpenMP
  nbody_cuda.cu     wariant CUDA (budowany z -DUSE_CUDA=ON)
  nbody_cuda_stub.cpp  zaślepka, gdy budujemy bez CUDA (np. macOS)
```

## Sprzęt pomiarowy

Wszystkie pomiary robimy na **jednym komputerze**, żeby porównanie CPU i GPU było uczciwe:

| | |
|---|---|
| CPU | AMD Ryzen 7 9800X3D: 8 rdzeni / 16 wątków, Zen 5 (AVX-512), 96 MB L3 |
| GPU | NVIDIA GeForce RTX 5070: Blackwell, compute capability 12.0 (`sm_120`) |
| System | Windows |

Konsekwencje dla projektu:
- **CUDA Toolkit ≥ 12.8.** Starsze wersje nie znają `sm_120`. Najlepiej zainstalować najnowszą (13.x).
- **`float` jako domyślny typ.** Karty GeForce liczą `double` ok. 64× wolniej niż `float`.
  Pomiar wersji GPU w `double` (`-DNBODY_USE_FLOAT=OFF`) może być ciekawym dodatkiem do sprawozdania.
- **OpenMP mierzymy dla 1, 2, 4, 8 i 16 wątków.** Powyżej 8 są już wątki SMT na tych samych rdzeniach,
  więc warto pokazać, ile to daje.
- **Dane mieszczą się w cache L3.** Nawet przy N = 65536 to tylko ok. 2,6 MB, więc wersja CPU jest
  ograniczona obliczeniami, a nie pamięcią RAM.

## Budowanie

### Windows z CUDA (komputer pomiarowy)

1. Zainstaluj **Visual Studio 2022 Build Tools** z obciążeniem *Desktop development with C++*.
   nvcc na Windowsie działa tylko z MSVC, nie z MinGW.
2. Zainstaluj **CUDA Toolkit** (≥ 12.8, najlepiej najnowszy) i aktualny sterownik NVIDIA.
3. W CLion:
   - *Settings → Build, Execution, Deployment → Toolchains*: dodaj **Visual Studio** i przesuń go na górę listy,
   - *Settings → … → CMake*: dodaj profil **Release** z opcją `-DUSE_CUDA=ON`.
4. Zbuduj i uruchom `nbody` z argumentem `--mode test`. Test nr 4 powinien pokazać nazwę karty.

Z linii poleceń (*x64 Native Tools Command Prompt for VS*):

```bat
cmake -S . -B build -DUSE_CUDA=ON
cmake --build build --config Release
build\Release\nbody.exe --mode test
```

Pomiary CPU i GPU róbcie z **tej samej** kompilacji (MSVC). Wyniki z różnych kompilatorów
(MSVC i MinGW/GCC) nie są wprost porównywalne.

### Bez GPU (macOS / Linux / Windows z MinGW)

```bash
cmake -S . -B build
cmake --build build --config Release
```

**macOS:** OpenMP wymaga `brew install libomp`. CMake znajdzie go sam.

### Opcje CMake

| opcja | domyślnie | znaczenie |
|---|---|---|
| `USE_CUDA` | `OFF` | buduje wariant GPU |
| `NBODY_USE_FLOAT` | `ON` | `float` zamiast `double` |
| `NBODY_NATIVE` | `ON` | instrukcje wektorowe procesora (`-march=native`, w MSVC `/arch:AVX2`) |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | architektura GPU; dla RTX 5070 to `120` |

## Uruchamianie

```bash
# testy poprawności (orbita 2 ciał, energia, pęd, OpenMP/CUDA vs seq)
./build/nbody --mode test          # albo: ctest --test-dir build

# benchmark wszystkich wariantów, kilka rozmiarów, z kontrolą poprawności
./build/nbody --mode all --n 1024,2048,4096 --steps 10 --repeats 5 --check

# skalowanie OpenMP względem liczby wątków, zapis do CSV
./build/nbody --mode omp --n 8192 --threads 1,2,4,8,16 --csv results/omp.csv
```

`./build/nbody --help` pokazuje wszystkie opcje.

### Kolumny CSV

| kolumna | znaczenie |
|---|---|
| `median_s`, `min_s`, `max_s`, `mean_s` | czas całej symulacji (`steps` kroków) |
| `ms_per_step` | mediana / liczba kroków |
| `ginteractions_per_s` | N² · kroki / czas, w miliardach |
| `energy_rel_drift` | względna zmiana energii (tylko z `--check`) |
| `max_diff_vs_seq` | maks. różnica położeń względem wersji sekwencyjnej (`--check`) |
| `cuda_h2d_s`, `cuda_kernel_s`, `cuda_d2h_s` | transfer na GPU, obliczenia, transfer z GPU |

## Ważne decyzje

- **Ten sam kod fizyki wszędzie:** `accel_on_body()` w `simulation.hpp` wołają
  wszystkie trzy warianty (w CUDA jako funkcja `__device__`).
- **Softening `eps > 0`:** człon i == j daje wtedy 0, więc pętla nie ma `if`.
- **Leapfrog (kick-drift-kick):** dobrze zachowuje energię; testy to sprawdzają.
- **Własny generator liczb z `mt19937_64`:** identyczne dane wejściowe na każdym systemie.
- **`-fno-math-errno`:** bez tej flagi GCC nie wektoryzował pętli sił w wersji
  OpenMP i ta na 1 wątku była ~2× wolniejsza od sekwencyjnej. Warto o tym
  wspomnieć w sprawozdaniu.
- **Brak `-ffast-math`:** wyniki CPU i GPU mają być porównywalne.

## Następne kroki

1. Zbudować z `-DUSE_CUDA=ON` na komputerze z NVIDIA i uruchomić `--mode test`.
2. Napisać kernel z tilingiem w pamięci `__shared__` i porównać z wersją podstawową.
3. Profilowanie: `nsys profile ./build/nbody --mode cuda --n 65536`, `ncu`.
4. Seria pomiarów (N = 256 … 65536, różna liczba wątków) do CSV.
5. Skrypt w Pythonie do wykresów: czas, przyspieszenie, efektywność, udział transferów.
