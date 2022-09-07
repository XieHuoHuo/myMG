#ifndef MG
#define MG
/**
 * @brief header file for myMultGrid.c
 * 
 */

void multigrid(double *u, double *f, int n, double w, int maxIter, double tol, int verbose, int coarselevel, int outersolver, int numSweeps);
void matvec(double *u, double *Au, int n);

#endif