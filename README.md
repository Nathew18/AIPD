# Symulacja grawitacyjnego problemu N-ciał

Projekt na przedmiot Akceleracja i obliczanie danych. Porównuje bezpośrednią
metodę O(N²) z integratorem leapfrog (kick-drift-kick) w trzech wariantach:

| Tryb | Implementacja |
|---|---|
| `seq` | CPU sekwencyjnie, punkt odniesienia |
| `omp` | CPU + OpenMP, równoległa pętla po ciałach |
| `cuda-naive` | CUDA, jeden wątek na ciało, dane z pamięci globalnej |

`all` uruchamia wszystkie dostępne warianty. Bez CUDA pomija wariant GPU.
Bez OpenMP `omp` działa na jednym wątku. Dane mają układ SoA; wszystkie warianty
używają tego samego wzoru siły, softeningu oraz identycznych warunków początkowych.

## Budowanie i testy

Wymagania: CMake ≥ 3.24, kompilator C++20. CUDA jest opcjonalna.
Na Windows do CUDA potrzebny jest MSVC oraz CUDA Toolkit obsługujący kartę.
Dla komputera pomiarowego (Ryzen 7 9800X3D, RTX 5070) używamy architektury `120`.

Windows, **x64 Native Tools Command Prompt for VS 2022**:

```bat
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
build\Release\nbody.exe --mode test
```

Bez GPU (Linux, macOS, Windows):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Na macOS OpenMP wymaga `brew install libomp`.
W CLion wybierz toolchain Visual Studio do CUDA oraz profil Release.

Jeżeli wykrywanie kompilatora MSVC kończy się błędem C1902, sprawdzona alternatywa
to generator Ninja z wykrywaniem kompilatorów w konfiguracji Release
(w terminalu narzędzi Visual Studio, z Ninja w PATH):

```bat
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-ninja
ctest --test-dir build-ninja --output-on-failure
```

| Opcja CMake | Domyślnie | Znaczenie |
|---|---|---|
| `USE_CUDA` | `OFF` | budowanie wariantu CUDA naive |
| `NBODY_USE_FLOAT` | `ON` | `float`; ustaw `OFF` dla `double` |
| `NBODY_NATIVE` | `ON` | instrukcje CPU: `-march=native` lub MSVC `/arch:AVX2` |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | architektura GPU; można ustawić jawnie |
| `BUILD_TESTING` | `ON` | testy CTest |

Testy obejmują orbitę dwóch ciał, energię, pęd, zgodność CPU/OpenMP/kernela
CUDA oraz pełny stan (pozycje, prędkości, przyspieszenia). Przypadki brzegowe:
N = 0, 1, 2, 31, 255, 256, 257, 513 i 0/1/7 kroków. Testy GPU wykonują się,
gdy karta jest dostępna; brak GPU jest jawnie zgłaszany. Python 3.10+ dodaje
testy CLI/CSV oraz analizy pomiarów. CI sprawdza CPU na Windows i Linux,
we float i double, oraz wariant bez OpenMP.

## Uruchamianie

Przykłady dla Windows (Linux/macOS: zastąp ścieżkę programu przez `./build/nbody`):

```bat
build\Release\nbody.exe --mode all --n 256,1024,4096 --threads 1,2,4,8,16 --steps 10 --repeats 5 --check --csv results\benchmark.csv
build\Release\nbody.exe --mode cuda-naive --n 65536 --steps 100
build\Release\nbody.exe --help
```

Opcje: `--mode`, `--n`, `--threads`, `--steps`, `--repeats`, `--warmup`,
`--seed`, `--dt`, `--eps`, `--check`, `--csv`.
Domyślnie: N=1024, 10 kroków, 5 powtórzeń, 1 rozgrzewka, seed=42,
dt=0.001, eps=0.01. Liczby ciał, wątków, kroków i powtórzeń muszą być dodatnie;
rozgrzewka może wynosić zero. Parametry fizyczne muszą być skończone.
Zbyt mały softening dla wybranego typu jest odrzucany.

Program tworzy katalog docelowego CSV i dopisuje wiersze bez powielania nagłówka.
Odrzuca dopisywanie do starszego formatu CSV. Informacje diagnostyczne trafiają
na stderr, więc stdout pozostaje poprawnym CSV. `--check` dodatkowo oblicza
energię i wynik referencyjny seq poza pomiarem czasu.

## Seria pomiarów i wykresy

Skrypty wymagają Pythona 3.10+. Tylko wykresy wymagają matplotlib:

```bat
python -m pip install -r requirements-analysis.txt
python scripts\benchmark.py build\Release\nbody.exe --n 256,512,1024,2048,4096,8192,16384 --threads 1,2,4,8,16 --steps 10 --repeats 5 --csv results\benchmark.csv
python scripts\analyze.py results\benchmark.csv --output results\analysis
```

Powstają `summary.csv` oraz `comparison.png`: czas, przyspieszenie,
efektywność OpenMP i udział transferów CUDA. Opcja `--no-plots` zapisuje tylko
podsumowanie CSV i działa bez matplotlib. Skrypt serii pomiarowej wymaga nowej
ścieżki CSV, aby nie mieszać eksperymentów.

Analiza łączy wyłącznie pomiary z identycznymi real/steps/seed/dt/eps.
Przyspieszenie wymaga seq z tym samym N; przy braku takiej bazy pole pozostaje
puste. Wiele pomiarów tej samej konfiguracji agreguje medianą ich median.
Dla N=32768/65536 można zmniejszyć liczbę kroków, ale **wszystkie porównywane
warianty dla danego N muszą mieć tę samą liczbę kroków**. Różne konfiguracje
analizuj oddzielnymi poleceniami. Pomiary porównuj w jednej kompilacji Release,
przy nieobciążonym komputerze; testy poprawności uruchom wcześniej.

## Znaczenie wyników

| Kolumny | Znaczenie |
|---|---|
| `mode,real,n,steps,threads,repeats` | wariant, precyzja i rozmiar zadania; threads dotyczy CPU |
| `median_s,min_s,max_s,mean_s` | czas całej symulacji, bez kopiowania stanu początkowego na CPU |
| `ms_per_step` | mediana czasu całości / liczba kroków |
| `ginteractions_per_s` | N² × (steps + 1) / czas / 10⁹, wlicza początkowy przebieg sił |
| `energy_rel_drift` | względna zmiana energii; pusta bez check lub gdy energia początkowa = 0 |
| `max_diff_vs_seq` | maksymalna odległość pozycji od seq po ostatnim powtórzeniu; tylko z check |
| `cuda_h2d_s,cuda_kernel_s,cuda_d2h_s` | średnie czasy faz GPU z cudaEvent; bez rozgrzewek |
| `seed,dt,eps` | parametry potrzebne do odtworzenia eksperymentu |

Całkowity czas CUDA obejmuje również alokację/zwalnianie pamięci i obsługę
wywołań na CPU. Czasy faz z cudaEvent nie muszą sumować się do czasu całości.
Dane trafiają na GPU raz, wszystkie kroki liczone są na karcie, wynik wraca raz.
Udział transferów w analizie = (średni H2D + średni D2H) / średni czas całości.

## Struktura i dalsza praca

- `include/`: stan SoA, wspólny wzór siły, interfejs i timer.
- `src/`: inicjalizacja, fizyka, CLI i trzy implementacje: seq, omp, cuda-naive.
- `tests/`: testy stanu, argumentów, CSV i obliczeń analitycznych.
- `scripts/`: seria pomiarów oraz analiza i wykresy.
- `PLAN.md`: stan realizacji i zadania eksperymentalne do sprawozdania.

Profilowanie, gdy dostępne są narzędzia NVIDIA:

```bat
nsys profile -o nbody-profile build\Release\nbody.exe --mode cuda-naive --n 65536 --steps 100 --repeats 1
ncu --set full build\Release\nbody.exe --mode cuda-naive --n 65536 --steps 10 --repeats 1 --warmup 0
compute-sanitizer --tool memcheck --error-exitcode 1 build\Release\nbody_tests.exe
```
