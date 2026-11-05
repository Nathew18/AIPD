# Plan projektu: Symulacja grawitacyjnego problemu N-ciał

Porównanie wydajności trzech implementacji tego samego algorytmu (bezpośrednia metoda O(N²)):
**CPU sekwencyjnie**, **CPU + OpenMP** i **GPU + CUDA**.

> **Stan wyjściowy:** wszystkie pliki `.hpp`, `.cpp` i `.cu` są puste. `CMakeLists.txt` ma błąd: wskazuje na `main.cpp` w katalogu głównym, a plik leży w `src/`. Szkielet katalogów jest dobry, ale praca zaczyna się praktycznie od zera.

---

## Sprzęt pomiarowy

Wszystkie trzy wersje mierzymy na jednym komputerze z Windowsem:
- **CPU:** AMD Ryzen 7 9800X3D (8 rdzeni / 16 wątków, Zen 5, AVX-512),
- **GPU:** NVIDIA GeForce RTX 5070 (Blackwell, `sm_120`).

Co z tego wynika:
- potrzebny jest **CUDA Toolkit ≥ 12.8** i kompilator **MSVC** (Visual Studio Build Tools), bo nvcc nie działa z MinGW,
- domyślnie liczymy we `float`, bo GeForce liczy `double` ok. 64× wolniej,
- OpenMP mierzymy dla 1, 2, 4, 8 i 16 wątków (powyżej 8 to wątki SMT).

Szczegóły konfiguracji są w `README.md`.

---

## Kolejność pracy

### Krok 0: build (`CMakeLists.txt`)

- [ ] Poprawić ścieżki do plików w `src/`.
- [ ] Dodać `find_package(OpenMP)` i podlinkować `OpenMP::OpenMP_CXX`.
- [ ] Zrobić CUDA opcjonalną: `option(USE_CUDA ...)` + `enable_language(CUDA)`. Wtedy projekt zbuduje się na Macu bez GPU.
- [ ] Zmienić `cmake_minimum_required(VERSION 4.1)` na `3.18`. Wersja 4.1 wymaga bardzo nowego CMake, a od 3.18 CUDA jest dobrze wspierana.

### Krok 1: `include/body.hpp`, czyli model danych

- [ ] Stałe:
  - `G`,
  - krok czasowy `dt`,
  - softening `eps`: mała wartość dodawana do odległości, żeby uniknąć dzielenia przez ~0, gdy dwa ciała są bardzo blisko.
- [ ] Układ danych **SoA** (Structure of Arrays): osobne tablice `x[]`, `y[]`, `z[]`, `vx[]`, `vy[]`, `vz[]`, `m[]` zamiast tablicy struktur. Na GPU łatwiej wtedy o wydajny dostęp do pamięci, a na CPU o wektoryzację.
- [ ] Zdecydować raz, `float` czy `double`, i używać tego typu we wszystkich wersjach. Karty konsumenckie liczą `double` wielokrotnie wolniej.

### Krok 2: `include/simulation.hpp`, czyli wspólny interfejs

- [ ] Generator warunków początkowych:
  - losowy, ze stałym `seed`, żeby każda wersja dostała identyczne dane,
  - opcjonalnie wczytywanie z pliku.
- [ ] Deklaracje trzech funkcji:
  - `step_seq(...)`,
  - `step_omp(...)`,
  - `step_cuda(...)`.
- [ ] Funkcje do sprawdzania poprawności:
  - całkowita energia (kinetyczna + potencjalna),
  - całkowity pęd.

> Ten interfejs trzeba ustalić **na samym początku**. Wtedy każdy członek zespołu pisze swoją część pod te same funkcje.

### Krok 3: `src/nbody_seq.cpp`, czyli wersja bazowa (najważniejszy etap)

- [ ] Podwójna pętla O(N²) liczy przyspieszenia.
- [ ] Aktualizacja prędkości i położeń. Użyjcie metody **leapfrog** albo **półjawnego Eulera**, bo dobrze zachowują energię.
- [ ] Testy poprawności:
  - dwa ciała na orbicie kołowej: orbita powinna się zamknąć,
  - energia całkowita nie powinna dryfować,
  - całkowity pęd powinien pozostać praktycznie stały.

> Ta wersja jest **wzorcem**. Wyniki OpenMP i CUDA porównujecie właśnie z nią.

### Krok 4: `include/benchmark.hpp` i `src/main.cpp`

- [ ] Timer na `std::chrono::steady_clock`.
- [ ] Jedno przebiegnięcie rozgrzewkowe, potem kilka pomiarów (np. 5) i **mediana**.
- [ ] Parametry z linii poleceń, np.:
  ```
  ./nbody --mode seq|omp|cuda --n 4096 --steps 100 --threads 8 --seed 42
  ```
- [ ] Wynik wypisywany w formacie **CSV**, żeby od razu dało się z niego robić wykresy.

### Krok 5: `src/nbody_openmp.cpp`

- [ ] `#pragma omp parallel for` na zewnętrznej pętli po `i`. Każdy wątek liczy przyspieszenia dla swoich ciał, więc nie ma konfliktów przy zapisie.
- [ ] Nie używać wersji z symetrią sił (3. zasada Newtona, liczenie każdej pary raz). W wersji równoległej powoduje wyścigi.
- [ ] Pomiary dla 1, 2, 4, 8 i 16 wątków (`omp_set_num_threads`).
- [ ] Sprawdzić, o ile pozycje różnią się od wersji sekwencyjnej. Powinny być prawie identyczne; drobne różnice wynikają z innej kolejności sumowania liczb zmiennoprzecinkowych.

### Krok 6: `src/nbody_cuda.cu`

- [ ] **Wersja 1 (prosta):** jeden wątek GPU liczy jedno ciało, a kernel `computeAccelerations` dostaje z pamięci globalnej dane wszystkich ciał.
- [ ] **Wersja 2 (zoptymalizowana):** tiling w pamięci współdzielonej (`__shared__`). Blok wątków wczytuje porcję ciał do szybkiej pamięci i każdy wątek liczy z nią oddziaływania. To klasyczna optymalizacja dla N-ciał.
- [ ] Dane kopiować na GPU **raz na początku**, wszystkie kroki wykonywać na karcie i kopiować wynik z powrotem na końcu.
- [ ] Mierzyć osobno czas kerneli i czas transferów (`cudaEvent`). Prezentacja zapowiada analizę kosztu transferu, więc to jest potrzebne.
- [ ] Profilowanie: Nsight Systems (`nsys`) i Nsight Compute (`ncu`).

### Krok 7: analiza wyników

- [ ] Rozmiary problemu: N = 256, 512, 1024 … 65536. Wersja sekwencyjna przy dużych N będzie bardzo wolna, więc dla niej zmniejszcie liczbę kroków.
- [ ] Metryki:
  - przyspieszenie `S = T_seq / T_par`,
  - efektywność OpenMP `E = S / liczba_wątków`,
  - opcjonalnie GFLOPS (ok. 20 operacji na jedną parę ciał).
- [ ] Wykresy w Pythonie (pandas + matplotlib) z plików CSV.
- [ ] Wnioski, np.:
  - od jakiego N GPU zaczyna wygrywać,
  - jak skaluje się OpenMP,
  - ile czasu zajmuje transfer danych.
- [ ] Dokumentacja końcowa.

---

## Proponowany podział na 4 osoby

Kroki 0–2 robicie razem, potem pracujecie równolegle:

| Osoba | Zakres |
|-------|--------|
| 1 | Wersja sekwencyjna i testy poprawności |
| 2 | OpenMP i pomiary skalowalności |
| 3 | CUDA i profilowanie |
| 4 | Benchmark, `main.cpp`, skrypty do wykresów, dokumentacja |

---

## Docelowa struktura plików

```
AIPD-main/
├── CMakeLists.txt
├── include/
│   ├── body.hpp          # dane ciał (SoA), stałe fizyczne
│   ├── simulation.hpp    # interfejs: init, step_seq/omp/cuda, energia, pęd
│   └── benchmark.hpp     # pomiar czasu, zapis CSV
└── src/
    ├── main.cpp          # parsowanie argumentów, uruchamianie trybów
    ├── nbody_seq.cpp     # wersja sekwencyjna (wzorzec)
    ├── nbody_openmp.cpp  # wersja wielowątkowa CPU
    └── nbody_cuda.cu     # kernel(e) GPU + zarządzanie pamięcią
```
