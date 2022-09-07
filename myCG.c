#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "myCG.h"
#include "myMultGrid.h"

#define MYCG_DEBUG 0

#if MYCG_DEBUG
// matrix-vector multiplication
// A is a 1-D Laplacian matrix
void matvec(double *u, double *Au, int n)
{
    int i;
    // Au[0] = 2*u[0] - u[1];
    for (i=1; i<n-1; i++)
        Au[i] = -u[i-1] + 2*u[i] - u[i+1];
    // Au[n-1] = -u[n-2] + 2*u[n-1];
    // divide by h^2
    // double h = 1.0/(n-1);
    // for (i=0; i<n; i++)
    //     Au[i] /= h*h;

}
#endif

void vecUpluseaV(double *u, double *v, int n, double alpha)
{
    for (int i = 0; i < n; i++) {
        u[i] += alpha * v[i];
    }
}
void vecaUpluseV(double *u, double *v, int n, double alpha)
{
    for (int i = 0; i < n; i++) {
        u[i] = alpha * u[i] + v[i];
    }
}
double vecInnProduct(double *u, double *v, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += u[i] * v[i];
    }
    return sum;
}
double vecNorm(double *u, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += u[i] * u[i];
    }
    return sqrt(sum);
}
double vecCopy(double *u, double *v, int n, double alpha)
{
    for (int i = 0; i < n; i++) {
        u[i] = alpha * v[i];
    }
}
void printVec(double *u, int n)
{
    for (int i = 0; i < n; i++) {
        printf("%f ", u[i]);
    }
    printf("\n");
}
#define CG_DEBUG 1
/**
 * @brief conjugate gradient method
 *        user must provide the matrix-vector multiplication function as matvec()
 * @param u 
 * @param f 
 * @param n
 * @param maxIter 
 * @param tol 
 * @param verbose 
 */
void CGSolver(double *u, double *f, int n, int maxIter, double tol, int verbose)
{
    // initialize
    double *r, *p, *Ap;
    r = (double *)malloc(n * sizeof(double));
    p = (double *)malloc(n * sizeof(double));
    Ap= (double *)malloc(n * sizeof(double));
    double *debugtmp = (double *)malloc(n * sizeof(double));
    matvec(u, r, n);
    vecaUpluseV(r, f, n, -1.0); // r = f - Au
    double rTr = vecInnProduct(r, r, n);
    double rTr_old = rTr;
    double res = sqrt(rTr);
    // printf("CG: initial residual = %e\n", res);
    int iter = 0;
    while ( (iter < maxIter) && (res > tol) )
    {
        iter++;
        if (iter == 1) {
            vecUpluseaV(p, r, n, 1.0); // p = r
        } else {
            double beta = rTr / rTr_old;
            vecaUpluseV(p, r, n, beta); // p = r + beta * p
        }
        matvec(p, Ap, n);
        double alpha = rTr / vecInnProduct(p, Ap, n);
        vecUpluseaV(u, p, n, alpha); // u = u + alpha * p
        // printVec(r, n);
        // printVec(Ap, n);
        vecUpluseaV(r, Ap, n, -alpha); // r = r - alpha * Ap
        rTr_old = rTr;
        rTr = vecInnProduct(r, r, n);
        res = sqrt(rTr);
        // if (verbose > 2) {
            printf("CGiter = %d, res = %e \n", iter, res);
        // }
    }
    
}

#if MYCG_DEBUG
#define PI 3.14159265358979323846
// right-hand side
double rhs(double x)
{
    return PI*PI*sin(PI*x);
}
// exact solution
double exact(double x)
{
    return sin(PI*x);
}
int main(int argc, char *argv[])
{
    int n = 129;
    double *u, *f;
    u = (double *)malloc(n * sizeof(double));
    f = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        u[i] = 0.0;
        f[i] = rhs(i*1.0/(n-1));
    }
    CGSolver(u, f, n, 1000, 1e-10, 1);
    // BiCGStabSolver(u, f, n, 1000, 1e-6, 1);
    return 0;
}
#endif