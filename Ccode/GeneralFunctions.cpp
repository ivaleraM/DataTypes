#include "GeneralFunctions.h"

// Transformations
/*double f(double x, double w){
    return logFun(1+w*gsl_sf_exp(x));
    
}*/
//inverse transformations f^-1
double fre_1(double x, double w, double mu){
    //w=1;
    return w*(x-mu);
}
double f_1(double x, double w){
    //w=1;
    return logFun(gsl_sf_exp(w*x)-1);
}

double fint_1(double x, double w, double theta_L, double theta_H){
    return -1/w*logFun((theta_H-x)/(x-theta_L));   
}

//derivative of the inverse transformations df^-1(x)/dx
double dfre_1(double x, double w){
    //return w;
    //return logFun(abs(w));  
    return w;
}

double df_1(double x, double w){
    //return w/(gsl_sf_exp(w*x)-1);
    return w/(1-gsl_sf_exp(-w*x)); 
}


double dfint_1(double x, double w, double theta_L, double theta_H){
    return 1/w*(theta_H-theta_L)/((theta_H-x)*(x-theta_L)); 
}

double xpdf_re(double x, double w, double muX, double mu, double s2Y, double s2u){
    //return logFun(gsl_ran_gaussian_pdf(f_1(x,w)-mu, sqrt(s2Y+s2u))) + logFun(abs(df_1(x,w)));
    //printf("dfre_1= %f ", dfre_1(x,w));
    return logFun(gsl_ran_gaussian_pdf(fre_1(x,w,muX)-mu, sqrt(s2Y+s2u)))+ logFun(fabs(dfre_1(x,w)));    
}
double xpdf_pos(double x, double w, double mu, double s2Y, double s2u){
    //return logFun(gsl_ran_gaussian_pdf(f_1(x,w)-mu, sqrt(s2Y+s2u))) + logFun(abs(df_1(x,w)));
    return logFun(gsl_ran_gaussian_pdf(f_1(x,w)-mu, sqrt(s2Y+s2u)))+ logFun(fabs(df_1(x,w)));
}
double xpdf_int(double x, double w, double theta_L, double theta_H, double mu, double s2Y, double s2u){
    //printf("dfint_1= %f ", logFun(fabs(dfint_1(x,w,theta_L,theta_H))));
    return logFun(gsl_ran_gaussian_pdf(fint_1(x,w,theta_L,theta_H)-mu, sqrt(s2Y+s2u)))+ logFun(fabs(dfint_1(x,w,theta_L,theta_H)));
}
double xpdf_dir(double x, double theta, double mu, double s2Y, double s2u){ 
    int n =100;//TODO
    return logFun(WNpdf(x,theta,mu,sqrt(s2Y+s2u),n));
}

double WNpdf(double x, double theta, double mu, double sigma, int n){
    //n=1;
    //sigma= theta/10;
    int kopt=(int)(mu/theta);
    //printf("mu=%f , kopt==%d ", mu, kopt);
    double px=0;
    for (int i=kopt-n; i<kopt+n+1; i++){
        px=px+gsl_ran_gaussian_pdf(x-mu+i*theta, sigma);
    }
    return px;
}


// Functions
double logFun(double x)
{
	if (x==0){return GSL_NEGINF;}
    else if(x<0){printf("Error: logarithm is not defined for negative numbers\n");return GSL_NEGINF;}
    else{return gsl_sf_log(x);}
}

double expFun(double x)
{
	if (x>300){return GSL_POSINF;}
    else if (x<-300){return 0;}
    else{return gsl_sf_exp(x);}
}

double maxArray (double x[],int D){
    double max=x[0];
    for (int c = 1; c <= D; c++){
        if(x[c]>max){max=x[c];}
    }
    return max;
}

int factorial (int N){
    int fact = 1;
    for (int c = 1; c <= N; c++){
        fact = fact * c;
    }
    return fact;
}

double max_vec (int N){
    int fact = 1;
    for (int c = 1; c <= N; c++){
        fact = fact * c;
    }
    return fact;
}

void matrix_multiply(gsl_matrix *A,gsl_matrix *B,gsl_matrix *C,double alpha,double beta,CBLAS_TRANSPOSE_t TransA,CBLAS_TRANSPOSE_t TransB) {
    /* Las matrices tienen que estar ordenadas por filas */
    
    /* C = \alpha op(A) op(B) + \beta C */
    
    gsl_blas_dgemm(TransA, TransB, alpha, A, B, beta, C);
    
}


double *column_to_row_major_order(double *A,int nRows,int nCols) {
    
    //mxArray *copy_mx = mxCreateDoubleMatrix(nRows,nCols,mxREAL);
    double *copy = (double *)malloc(nRows*nCols*sizeof(double));
    if (NULL==copy) {
        mexErrMsgTxt("Not enough memory (in column_to_row_major_order)\n");
	}
    
    for (int i=0; i<nRows*nCols; i++) {
        int cociente = i/nRows;
        int x = cociente+nCols*(i-nRows*cociente);
        copy[x] = A[i];
	}
    
    return copy;
}

// double *row_to_column_major_order(double *A,int nRows,int nCols) {
//     
//     //mxArray *copy_mx = mxCreateDoubleMatrix(nRows,nCols,mxREAL);
//     double *copy = (double *)malloc(nRows*nCols*sizeof(double));
//     if (NULL==copy) {
//         mexErrMsgTxt("Not enough memory (in row_to_column_major_order)\n");
// 	}
//     
//     for (int i=0; i<nRows*nCols; i++) {
//         int cociente = i/nRows;
//         int x = cociente+nCols*(i-nRows*cociente);
//         copy[x] = A[i];
// 	}
//     
//     return copy;
// }


double det_get(gsl_matrix *Amat, int Arows, int Acols, int inPlace) {
/*
  inPlace = 1 => A is replaced with the LU decomposed copy.
  inPlace = 0 => A is retained, and a copy is used for LU.
*/
   
   double det;
   int signum;
   gsl_permutation *p = gsl_permutation_alloc(Arows);
   gsl_matrix *tmpA;

   if (inPlace)
      tmpA = Amat;
   else {
     tmpA = gsl_matrix_alloc(Arows, Acols);
     gsl_matrix_memcpy(tmpA , Amat);
   }


   gsl_linalg_LU_decomp(tmpA, p, &signum);
   det = gsl_linalg_LU_det(tmpA, signum);
   gsl_permutation_free(p);
   if (! inPlace)
      gsl_matrix_free(tmpA);

   
   return det;
}

double lndet_get(gsl_matrix *Amat, int Arows, int Acols, int inPlace) {
/*
  inPlace = 1 => A is replaced with the LU decomposed copy.
  inPlace = 0 => A is retained, and a copy is used for LU.
*/
   
   double det;
   int signum;
   gsl_permutation *p = gsl_permutation_alloc(Arows);
   gsl_matrix *tmpA;

   if (inPlace)
      tmpA = Amat;
   else {
     tmpA = gsl_matrix_alloc(Arows, Acols);
     gsl_matrix_memcpy(tmpA, Amat);
   }


   gsl_linalg_LU_decomp(tmpA, p, &signum);
   det = gsl_linalg_LU_lndet(tmpA);
   gsl_permutation_free(p);
   if (! inPlace)
      gsl_matrix_free(tmpA);

   
   return det;
}


// gsl_matrix *inverse(gsl_matrix *Amat, int Asize)
// {
// 	// Define all the used matrices
//     int s;
// 	//gsl_matrix *inverse = gsl_matrix_alloc(Asize, Asize);
// 	//gsl_permutation *perm = gsl_permutation_alloc(Asize);
//     gsl_matrix *Amatcopy = gsl_matrix_alloc(Asize, Asize);
//     gsl_matrix_memcpy(Amatcopy, Amat);
//     
//     
// 	// Make LU decomposition of matrix m
// 	//gsl_linalg_LU_decomp(Amatcopy, perm, &s);
//     gsl_linalg_cholesky_decomp (Amatcopy);
//     
// 	// Invert the matrix m
// 	//gsl_linalg_LU_invert(Amatcopy, perm, inverse);
//     gsl_linalg_cholesky_invert (Amatcopy);
//     
//     //gsl_matrix_free(Amatcopy);
//     
//     return Amatcopy;
// }

void inverse(gsl_matrix *Amat, int Asize)
{
	// Make LU decomposition of matrix m
	//gsl_linalg_LU_decomp(Amatcopy, perm, &s);
    gsl_linalg_cholesky_decomp (Amat);
    
	// Invert the matrix m
	//gsl_linalg_LU_invert(Amatcopy, perm, inverse);
    gsl_linalg_cholesky_invert (Amat);
    
}


double trace(gsl_matrix *Amat, int Asize)
{
	// Assume Amat is square
    double resul=0;
    
    double *p = Amat->data;
    
    for(int i=0; i<Asize; i++) {
        resul += p[i*Asize+i];
	}
    
    return resul;
}

// Sampling functions
int poissrnd(double lambda) {
    double L = gsl_sf_exp(-lambda);
    int k = 0;
    double p = 1;
    do {
        k++;
        p *= drand48();
    } while( p > L);
    return (k-1);
}

int mnrnd(double *p, int nK){
    double pMin=0;
    double pMax=p[0];
    double s=drand48();
    int k=0;
    int flag=1;
    int Knew;
    while (flag){
        
        if (s>pMin & s<=pMax){flag=0; Knew=k;}
        else{pMin+=p[k];
            pMax+=p[k+1];
            }
        k++;
        
        }
    return Knew;
    }

void mvnrnd(gsl_vector *x, gsl_matrix *Sigma,gsl_vector *Mu, int K, const gsl_rng *seed){
    
    gsl_matrix *A = gsl_matrix_alloc(K, K);
    gsl_matrix_memcpy(A, Sigma);
    gsl_linalg_cholesky_decomp (A);
    
    for (int k=0; k<K; k++){
        gsl_vector_set (x, k, gsl_ran_ugaussian (seed));
        }
    
    gsl_blas_dtrmv (CblasLower, CblasNoTrans, CblasNonUnit, A, x);
    gsl_vector_add (x, Mu);
    
    gsl_matrix_free(A);
    
    }

double truncnormrnd(double mu, double sigma, double xlo, double xhi){
    
    //if (xlo>xhi){printf("error: xlo=%f >xhi=%f \n", xlo, xhi);}
    double plo=gsl_cdf_ugaussian_P((xlo-mu)/sigma);
    double phi=gsl_cdf_ugaussian_P((xhi-mu)/sigma);
    double r=drand48();
    r=plo+(phi-plo)*r;
    //printf("r= %f \n", r);
    double z=gsl_cdf_ugaussian_Pinv(r);
    //if (gsl_isnan(mu+z*sigma)){printf("mu=%f, xlo= %f, xhi= %f \n",mu, xlo,  xhi);}
    double x=mu+z*sigma;
//     if(x<xlo){x=xlo;}
//     if(x>xhi){x=xhi;}
    return x;
    
    }
