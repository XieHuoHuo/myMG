/**
 * @file myMultGrid.c
 * @author Xie Yan (xieyan2021@lsec.cc.ac.cn)
 * @brief single thread version of multigrid solver for Poisson equation on a 2d-square domain [0,1]x[0,1]
 *        with Dirichlet boundary conditions, using uniform grid and weighted-Jacobi smoother
 * @version 0.1
 * @date 2022-09-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#define DEBUG 0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "myMultGrid.h"
#include "myCG.h"

/**
 * @brief print the dofs of the grid
 * 
 */
void printGrid(double *u, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            // printf("%f ", u[i * n + j]);
            printf("[%d,%d]:%f ", i, j, u[i * n + j]);
        }
        printf("\n");
    }
}
// construct analytical functions
#define PI 3.14159265358979323846

// solution function
double sol(double x, double y) {
    // return sin(PI * x) * sin(PI * y);
    return x * x + y * y;
}
// right-hand side function
double rhs(double x, double y) {
    // return 2 * PI * PI * sin(PI * x) * sin(PI * y);
    return 4.0;
}

/**
 * @brief matrix-free matrix-vector multiplication
 *        it can be used in CG solver
 */
void matvec(double *u, double *v, int n)
{
    // here n is the number of dofs
    // one need to convert n to mesh size by sqrt(n)
    n = (int)sqrt(n);
    int i, j;
    double h = 1.0 / (n - 1);
    double h2 = h * h;
    for (i = 1; i < n - 1; i++)
    {
        for (j = 1; j < n - 1; j++)
        {
            v[i * n + j] = - (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1] - 4 * u[i * n + j]) / h2;
        }
    }
}

// vector u = u - v
void vecUminusV(double *u, double *v, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        u[i] -= v[i];
    }
}
/**
 * @brief weighted-Jacobi smoother
 * 
 */
void wJacobi(double *u, double *f, int n, double w)
{
    double h = 1.0 / (n - 1);
    double h2 = h * h; // h^2
    // update interior points
    double *tmpu;
    tmpu = (double *)malloc(n * n * sizeof(double));
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            tmpu[i * n + j] = w * ( h2*f[i * n + j] + u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1] ) / 4.0
                         + (1 - w) * u[i * n + j];
        }
    }
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            u[i * n + j] = tmpu[i * n + j];
        }
    }
    free(tmpu);
}

/**
 * @brief restriction operator
 *  used to restrict the residual from fine grid to coarse grid
 *  weighted average of nine points:
 *      1/4 1/2 1/4
 *      1/2 1.0 1/2
 *      1/4 1/2 1/4
 *  reduce No. grid-point to its 1/4 (approx)
 *      * = * = * = * = *
 *      = - - - - - - - =
 *      * - + - + - + - *
 *      = - - - - - - - =
 *      * - + - + - + - *
 *      = - - - - - - - =
 *      * = * = * = * = *
 *  u is of size (2*m-1) * (2*m-1), ru is of size m * m
 */
void restriction(double *u, double *ru, int n)
{
    if (n/2 < 1) {
        printf("Error: n/2 < 2 in restriction()! n = %d is too small! \n ", n);
        exit(1);
    }
    if (n%2 == 0) {
        printf("Error: n is even in restriction()! n = %d is not allowed! \n ", n);
        exit(1);
    }
    int m = (n + 1) / 2;
    // interior points
    for (int i = 1; i < m - 1; i++) {
        for (int j = 1; j < m - 1; j++) {
            ru[i * m + j] = 1.0 * u[2 * i * n + 2 * j] 
                          + 0.5  * (u[(2 * i - 1) * n + 2 * j] 
                          +         u[(2 * i + 1) * n + 2 * j] 
                          +         u[2 * i * n + 2 * j - 1] 
                          +         u[2 * i * n + 2 * j + 1])
                          + 0.25 * (u[(2 * i - 1) * n + 2 * j - 1] 
                          +         u[(2 * i - 1) * n + 2 * j + 1] 
                          +         u[(2 * i + 1) * n + 2 * j - 1] 
                          +         u[(2 * i + 1) * n + 2 * j + 1]);
            ru[i * m + j] /= 4.0;
        }
    }
    // boundary points (only need to set zero)
    for (int i = 0; i < m; i++) {
        ru[i * m] = 0.0;
        ru[i * m + m - 1] = 0.0;
        ru[i] = 0.0;
        ru[(m - 1) * m + i] = 0.0;
    }
}

/**
 * @brief prolongation operator
 *  inverse operation of restriction operator,
 *  used to prolong the correction from coarse grid to fine grid
 *  scatter coarse-grid dofs to fine-grid dofs with dual weights:
 *      1/4 1/2 1/4
 *      1/2 1.0 1/2
 *      1/4 1/2 1/4
 *  increase No. grid-point to its 4 times
 *  u is of size m * m, pu is of size (2*m-1) * (2*m-1)
 */  
void prolongation(double *u, double *pu, int m)
{
    int n = 2 * m - 1;
    for (int i = 0; i < n * n; i++) {
        pu[i] = 0.0;
    }
    // interior points
    for (int i = 1; i < m - 1; i++) {
        for (int j = 1; j < m - 1; j++) {
            pu[ 2 * i      * n + 2 * j]     += 1.0  * u[i * m + j];
            pu[(2 * i - 1) * n + 2 * j]     += 0.5  * u[i * m + j];
            pu[(2 * i + 1) * n + 2 * j]     += 0.5  * u[i * m + j];
            pu[ 2 * i      * n + 2 * j - 1] += 0.5  * u[i * m + j];
            pu[ 2 * i      * n + 2 * j + 1] += 0.5  * u[i * m + j];
            pu[(2 * i - 1) * n + 2 * j - 1] += 0.25 * u[i * m + j];
            pu[(2 * i - 1) * n + 2 * j + 1] += 0.25 * u[i * m + j];
            pu[(2 * i + 1) * n + 2 * j - 1] += 0.25 * u[i * m + j];
            pu[(2 * i + 1) * n + 2 * j + 1] += 0.25 * u[i * m + j];
        }
    }
}

/**
 * @brief compute residual
 * 
 */
void residual(double *u, double *f, double *r, int n)
{
    double h = 1.0 / (n - 1);
    double h2 = h * h; // h^2
    // update interior points
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            r[i * n + j] = f[i * n + j] + (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1] - 4 * u[i * n + j]) / h2;
        }
    }
    // set boundary points to zero
    for (int i = 0; i < n; i++) {
        r[i] = 0.0;
        r[i * n] = 0.0;
        r[i * n + n - 1] = 0.0;
        r[(n - 1) * n + i] = 0.0;
    }
}

/**
 * @brief compute error
 * 
 */
void error(double *u, double *err, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            err[i * n + j] = u[i * n + j] - sol(i * 1.0 / (n - 1), j * 1.0 / (n - 1));
        }
    }
}

/**
 * @brief multigrid solver for Poisson equation on a 2d-square domain with Dirichlet boundary conditions
 *        use V-cycle by default
 * @param u solution vector
 * @param f right-hand side vector
 * @param n size of u and f
 * @param w weight of weighted-Jacobi smoother
 * @param maxIter maximum number of iterations
 * @param tol tolerance of residual
 * @param verbose print residual every verbose iterations
 *  
 */
#define USEJACOBI 0
void multigrid(double *u, double *f, int n, double w, int maxIter, double tol, int verbose, int coarselevel, int outersolver, int numSweeps)
{
    if ( (coarselevel < 1) || (n < 3) ) {
        #if USEJACOBI
        if (verbose > 2)
        printf(">>> doing weighted-Jacobi smoothing on the coarsest level \n");
        // do weighted-Jacobi iteration to solve the problem
        double *err = (double *)malloc(n * n * sizeof(double));
        for (int iter = 0; iter < 100; iter++) {
            // error(u, err, n);
            // printGrid(err, n);
            wJacobi(u, f, n, w);
        }
        if (verbose > 2)
        printf("<<< done with weighted-Jacobi smoothing on the coarsest level \n");
        #else
        // do CG iteration to solve the problem
        if (verbose > 2)
        printf(">>> doing CG iteration on the coarsest level \n");
        // BiCGStabSolver(u, f, n*n, 10, tol, verbose);
        CGSolver(u, f, n*n, 100, 1e-5, verbose);
        if (verbose > 2)
        printf("<<< done with CG iteration on the coarsest level \n");
        #endif
        if (verbose) {
            // compute residual
            double *r = (double *)malloc(n * n * sizeof(double));
            residual(u, f, r, n);
            // printGrid(u, n);
            double res = 0.0;
            for (int i = 0; i < n * n; i++) {
                res += r[i] * r[i];
            }
            res = sqrt(res);
            // printf("res = %e \n", res);
            free(r);    
        }
        return;
    }

    if (verbose > 2)
    printf(">>> multigrid solver: n = %d, w = %f, maxIter = %d, tol = %f, verbose = %d, coarselevel = %d \n", n, w, maxIter, tol, verbose, coarselevel);
    int iter = 0;
    double res = 1.0;
    int m = (n + 1) / 2;
    double *r = (double *)malloc(n * n * sizeof(double));
    while ( (iter < maxIter) && (res > tol) )
    {
        iter++;
        // pre-smoothing
        for (int i = 0; i < numSweeps; i++)
        wJacobi(u, f, n, w);
        // compute residual
        residual(u, f, r, n);
        // restrict residual to coarse-grid
        double *rc = (double *)malloc(m * m * sizeof(double));
        restriction(r, rc, n);
        // solve on coarse-grid
        double *uc = (double *)malloc(m * m * sizeof(double));
        // set uc to zero as initial guess
        for (int i = 0; i < m * m; i++) {
            uc[i] = 0.0;
        }
        printGrid(uc, m);
        multigrid(uc, rc, m, w, 1, tol, verbose, coarselevel - 1, 0, numSweeps);
        // prolongate coarse-grid solution to fine-grid
        double *pu = (double *)malloc(n * n * sizeof(double));
        prolongation(uc, pu, m);
        // update fine-grid solution
        for (int i = 0; i < n * n; i++) {
            u[i] += pu[i];
        }
        // post-smoothing
        for (int i = 0; i < numSweeps; i++)
        wJacobi(u, f, n, w);

        if ( (verbose > 0) && (outersolver) ) {
            // compute residual
            residual(u, f, r, n);
            // double res = 0.0;
            res = 0.0;
            for (int i = 0; i < n * n; i++) {
                res += r[i] * r[i];
            }
            res = sqrt(res);
            printf("iter = %d, res = %e \n", iter, res);
        }
    }
    if (verbose > 2)
    printf("<<< done with multigrid solver \n");
}

#ifdef DEBUG

int main()
{
    // set up parameters
    int     N           = 17;      // N = 2**k + 1
    double  w           = 2.0 / 3.0;// weight of weighted-Jacobi smoother
    int     maxIter     = 40;       // maximum number of iterations
    double  tol         = 1e-6;     // tolerance of residual
    int     verbose     = 1;        // print residual every verbose iterations
    int     coarselevel = 2;        // number of coarse-grid levels
    int     numSweeps   = 3;        // number of sweeps on each level

    // initialize dofs
    double *u = (double *)malloc(N * N * sizeof(double));
    double *f = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) {
        u[i] = 0.0; // initial guess and zero boundary conditions
    }

    double h = 1.0 / (N - 1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            f[i * N + j] = rhs(i * h, j * h);
        }
    }
    // set boundary to zero
    for (int i = 0; i < N; i++) {
        f[i * N] = 0.0;
        f[i * N + N - 1] = 0.0;
        f[i] = 0.0;
        f[(N - 1) * N + i] = 0.0;
    }

    // solve Poisson equation and time it
    clock_t start = clock();
    printf(">>> solving Poisson equation on a 2d-square domain with Dirichlet boundary conditions \n");
    multigrid(u, f, N, w, maxIter, tol, verbose, coarselevel, 1, numSweeps);
    printf("<<< done with solving Poisson equation \n");
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("time = %f seconds \n", time);

    // f L2 norm
    double fL2 = 0.0;
    for (int i = 0; i < N * N; i++) {
        fL2 += f[i] * f[i];
    }
    fL2 = sqrt(fL2);
    printf("L2 norm of f = %e \n", fL2);

    // free memory
    free(u);
    free(f);

    return 0;
}
#endif