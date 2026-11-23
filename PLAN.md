# Plan projektu: symulacja grawitacyjnego problemu N-ciał

## Zrealizowany kod

- [x] CMake: opcjonalna CUDA, OpenMP i budowanie CPU bez obu technologii.
- [x] Układ SoA oraz wspólny typ real (float/double).
- [x] Deterministyczny generator warunków początkowych.
- [x] Wspólny wzór siły z softeningiem i integrator leapfrog KDK.
- [x] Implementacja sekwencyjna O(N²).
- [x] Implementacja OpenMP z podziałem po ciałach, bez współdzielonych zapisów.
- [x] CUDA podstawowa: jeden wątek na ciało, odczyty globalne.
- [x] Bezpieczna obsługa niepełnych bloków CUDA, pustych układów i błędów.
- [x] Jednorazowy transfer danych przed/po całej symulacji; osobne czasy faz.
- [x] Benchmark: rozgrzewka, powtórzenia, mediana/min/max/średnia, CSV.
- [x] Walidacja argumentów, metadane eksperymentu i kontrola formatu CSV.
- [x] Testy fizyki, pełnego stanu, granic bloków, CLI/CSV i analizy.
- [x] Skrypt serii pomiarowej i wykresy czasu, przyspieszenia, efektywności, transferów.
- [x] CI CPU: Windows/Linux, float/double, również bez OpenMP.
- [x] Instrukcje budowania, testów i pomiarów w README.

## Eksperymenty do sprawozdania

Poniższe zadania wymagają wykonania pomiarów i interpretacji; nie są brakami kodu.

- [ ] Pełna seria N = 256, 512, 1024 … 65536.
- [ ] OpenMP dla 1, 2, 4, 8 i 16 wątków.
- [ ] Porównanie seq, omp i CUDA naive oraz float/double w osobnych seriach.
- [ ] Profilowanie Nsight Systems / Nsight Compute.
- [ ] Wyznaczenie punktu, od którego GPU jest szybsze od CPU.
- [ ] Omówienie udziału transferów i kosztów stałych GPU.
- [ ] Interpretacja skalowania OpenMP, w tym wątków SMT.
- [ ] Końcowe sprawozdanie i prezentacja.

## Zasady pomiaru

Wszystkie warianty porównujemy na jednym komputerze, z tej samej kompilacji
Release i tymi samymi parametrami. Sprzęt docelowy: Ryzen 7 9800X3D + RTX 5070.
Dla dużych N można zmniejszyć liczbę kroków, zachowując ten sam parametr
w porównywanych implementacjach. Przyspieszenie S = T_seq / T_wariant;
efektywność OpenMP E = S / liczba_wątków.
