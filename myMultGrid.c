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
            printf("%f ", u[i * n + j]);
        }
        printf("\n");
    }
}
// construct analytical functions
#define PI 3.14159265358979323846

// solution function
double sol(double x, double y) {
    return sin(PI * x) * sin(PI * y);
    // return x * x + y * y;
}
// right-hand side function
double rhs(double x, double y) {
    return 2 * PI * PI * sin(PI * x) * sin(PI * y);
    // return 4.0;
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
}

/**
 * @brief restriction operator
 *  weighted average of nine points:
 *      1/4 1/2 1/4
 *      1/2 1.0 1/2
 *      1/4 1/2 1/4
 *  reduce No. grid-point to its 1/4
 *  u is of size (2*m+1) * (2*m+1), ru is of size m * m
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
    int m = (n - 1) / 2;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            ru[i * m + j] = 0.25 * u[2 * i * n + 2 * j] + 0.5 * u[(2 * i + 1) * n + 2 * j] + 0.25 * u[(2 * i + 2) * n + 2 * j]
                          + 0.5 * u[2 * i * n + 2 * j + 1] + u[(2 * i + 1) * n + 2 * j + 1] + 0.5 * u[(2 * i + 2) * n + 2 * j + 1]
                          + 0.25 * u[2 * i * n + 2 * j + 2] + 0.5 * u[(2 * i + 1) * n + 2 * j + 2] + 0.25 * u[(2 * i + 2) * n + 2 * j + 2];
            ru[i * m + j] /= 4.0;
        }
    }
}

/**
 * @brief prolongation operator
 *  inverse operation of restriction operator,
 *  scatter coarse-grid dofs to fine-grid dofs with dual weights:
 *      1/4 1/2 1/4
 *      1/2 1.0 1/2
 *      1/4 1/2 1/4
 *  increase No. grid-point to its 4 times
 *  u is of size m * m, pu is of size (2*m+1) * (2*m+1)
 */  
void prolongation(double *u, double *pu, int m)
{
    int n = 2 * m + 1;
    for (int i = 0; i < n * n; i++) {
        pu[i] = 0.0;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            pu[2 * i * (2 * m + 1) + 2 * j] += 0.25 * u[i * m + j];
            pu[(2 * i + 1) * (2 * m + 1) + 2 * j] += 0.5 * u[i * m + j];
            pu[(2 * i + 2) * (2 * m + 1) + 2 * j] += 0.25 * u[i * m + j];
            pu[2 * i * (2 * m + 1) + 2 * j + 1] += 0.5 * u[i * m + j];
            pu[(2 * i + 1) * (2 * m + 1) + 2 * j + 1] += u[i * m + j];
            pu[(2 * i + 2) * (2 * m + 1) + 2 * j + 1] += 0.5 * u[i * m + j];
            pu[2 * i * (2 * m + 1) + 2 * j + 2] += 0.25 * u[i * m + j];
            pu[(2 * i + 1) * (2 * m + 1) + 2 * j + 2] += 0.5 * u[i * m + j];
            pu[(2 * i + 2) * (2 * m + 1) + 2 * j + 2] += 0.25 * u[i * m + j];
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
void multigrid(double *u, double *f, int n, double w, int maxIter, double tol, int verbose, int coarselevel, int outersolver)
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
        CGSolver(u, f, n*n, 10, tol, verbose);
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
            printf("res = %f \n", res);
            free(r);    
        }
        return;
    }

    if (verbose > 2)
    printf(">>> multigrid solver: n = %d, w = %f, maxIter = %d, tol = %f, verbose = %d, coarselevel = %d \n", n, w, maxIter, tol, verbose, coarselevel);
    int iter = 0;
    while (iter < maxIter)
    {
        iter++;
        // pre-smoothing
        wJacobi(u, f, n, w);
        // compute residual
        double *r = (double *)malloc(n * n * sizeof(double));
        residual(u, f, r, n);
        // restrict residual to coarse-grid
        int m = (n - 1) / 2;
        double *rc = (double *)malloc(m * m * sizeof(double));
        restriction(r, rc, n);
        // solve on coarse-grid
        double *uc = (double *)malloc(m * m * sizeof(double));
        for (int i = 0; i < m * m; i++) {
            uc[i] = 0.0;
        }
        multigrid(uc, rc, m, w, 1, tol, verbose, coarselevel - 1, 0);
        // prolongate coarse-grid solution to fine-grid
        double *pu = (double *)malloc(n * n * sizeof(double));
        prolongation(uc, pu, m);
        // update fine-grid solution
        for (int i = 0; i < n * n; i++) {
            u[i] += pu[i];
        }
        // post-smoothing
        wJacobi(u, f, n, w);

        if ( (verbose > 0) && (outersolver) ) {
            // compute residual
            residual(u, f, r, n);
            double res = 0.0;
            for (int i = 0; i < n * n; i++) {
                res += r[i] * r[i];
            }
            res = sqrt(res);
            printf("iter = %d, res = %f \n", iter, res);
        }
    }
    if (verbose > 2)
    printf("<<< done with multigrid solver \n");
}

#ifdef DEBUG

int main()
{
    // set up parameters
    int N = 127; // N = 2**k - 1
    double w = 2.0 / 3.0; // weight of weighted-Jacobi smoother
    int maxIter = 1; // maximum number of iterations
    double tol = 1e-6; // tolerance of residual
    int verbose = 1; // print residual every verbose iterations

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

    // solve Poisson equation
    // double *err = (double *)malloc(N * N * sizeof(double));
    // error(u, err, N);
    // printGrid(err, N);
    multigrid(u, f, N, w, maxIter, tol, verbose, 2, 1);

    // exact solution
    // double *exactsol = (double *)malloc(N * N * sizeof(double));
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         exactsol[i * N + j] = sol(i * h, j * h);
    //     }
    // }
    // double *r;
    // r = (double *)malloc(N * N * sizeof(double));
    // residual(exactsol, f, r, N);
    // printGrid(r, N);
    return 0;
}
#endif