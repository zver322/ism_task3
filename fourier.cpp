#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include "array_types.hpp"

#include "mpi.h"

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;
const double PI = 3.141592653589793;

template <class T>
void fill_random(vec<T> x, T xmin, T xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

template <class T>
void fill_random_cos(vec<T> x, T ampl_max, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(-ampl_max, ampl_max);

    ptrdiff_t n = x.length();
    for (ptrdiff_t i = 0; i < n; i++)
    {
        x(i) = 0;
    }
    for (int imode=1; imode<=5; imode++)
    {
        T ampl = dist(rng);
        for (ptrdiff_t i = 0; i < n; i++)
        {
            x(i) += ampl * cos(imode * (2 * PI * i) / n);
        }
    }
}

// the algorithm assumes that x is synchronized between processes,
// only part of f is computed by each process
void cosine_dft(vec<double> f, vec<double> x, MPI_Comm comm)
{    
    ptrdiff_t i, k, nf = f.length(), nx = x.length();

    int myrank, comm_size;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &myrank);

    ptrdiff_t fchunk = nf / comm_size;
    ptrdiff_t kstart = fchunk * myrank;

    double omega = 0;
    for (k=kstart; k < kstart + fchunk; k++)
    {
        f(k) = 0;
    }
    for (i=0; i < nx; i++)
    {
        for (k=kstart; k < kstart + fchunk; k++)
        {
            omega = 2 * PI * k / nx;
            f(k) += x(i) * cos(omega * i);
        }
    }
    MPI_Allgather(f.raw_ptr()+kstart, fchunk, MPI_DOUBLE, f.raw_ptr(), fchunk, MPI_DOUBLE, comm);
}

// read an integer number from stdin into `n`
void read_integer(int* n, int rank, MPI_Comm comm)
{
    if (rank==0)
    {
        std::cin >> *n;
    }

    MPI_Bcast(n, 1, MPI_INT, 0, comm);
}

int main(int argc, char* argv[])
{
    int n;

    int myrank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    read_integer(&n, myrank, MPI_COMM_WORLD);

    vec<double> x(n);
    vec<double> f(n);

    if (myrank == 0)
    {
        fill_random_cos(x, 100.0, 9876);
    }

    MPI_Bcast(x.raw_ptr(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    cosine_dft(f, x, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (myrank == 0)
    {
        // f[end-5] - f[end-1] must be non-zero, the rest must be zero
        std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << "Timing: " << t1 - t0 << " sec\n";
        for (ptrdiff_t i=f.length()-10; i<f.length(); i++){
            std::cout << "f[" << i << "] = " << f(i) << '\n';
        }
    }
    MPI_Finalize();
    return 0;
}
