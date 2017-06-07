#include "Baseline.h"
#include "GeneralFunctions.cpp"

//*********************************INPUTS**************************//

#define input_X prhs[0]
#define input_C prhs[1]
#define input_R prhs[2]
#define input_paramW prhs[3]
#define input_s2Z prhs[4]
#define input_s2B prhs[5]
#define input_sY prhs[6]
#define input_s2u prhs[7]
#define input_s2theta prhs[8]
#define input_Nsim prhs[9]
#define input_maxK prhs[10]
#define input_XT prhs[11]

//*********************************OUTPUTS**************************//

#define output_K plhs[0]
#define output_W plhs[1]
#define output_countErr plhs[2]
#define output_LIK plhs[3]

void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) {
    
    //..................CHECKING INPUTS AND OUTPUTS.............//
    /* Las matrices vienen ordenadas por columnas */
    
    if (nrhs!=12) {
        mexErrMsgTxt("Invalid number of arguments\n");
    }
    
    const mwSize XNumDim = mxGetNumberOfDimensions(input_X);
    //const mwSize ZNumDim = mxGetNumberOfDimensions(input_Z);
    const mwSize s2BNumDim = mxGetNumberOfDimensions(input_s2B);
    
    const mwSize* Xdim = mxGetDimensions(input_X);
    //const mwSize* Zdim = mxGetDimensions(input_Z);
    const mwSize* Cdim = mxGetDimensions(input_C);
    const mwSize* Wdim = mxGetDimensions(input_paramW);
    const mwSize* Rdim = mxGetDimensions(input_R);
    
    int N = Xdim[0];
    int D = Xdim[1];
    //int K = Zdim[1];
    //printf("C1 %d, C2 %d \n",Cdim[0], Cdim[1]);
    if (Cdim[1]!=D & Cdim[0]!=1){
        mexErrMsgTxt("Invalid number of dimensions for vector C\n");
        }
    if (Rdim[1]!=D & Rdim[0]!=1){
        mexErrMsgTxt("Invalid number of dimensions for vector R\n");
        }
//     if (Wdim[0]!=D & Cdim[1]!=3){
//         mexErrMsgTxt("Invalid number of dimensions for vector W\n");
//         }
       
    double *X_dou = mxGetPr(input_X);
    double *XT_dou = mxGetPr(input_XT);
    //double *Z_dou = mxGetPr(input_Z);
    double *R_dou = mxGetPr(input_R);
    double *C_dou = mxGetPr(input_C);
    double *W_dou = mxGetPr(input_paramW);
    
    double s2B = mxGetScalar(input_s2B);
    double s2Y = mxGetScalar(input_sY);
    double s2Z = mxGetScalar(input_s2Z);
    double s2theta = mxGetScalar(input_s2theta);
    double s2u = mxGetScalar(input_s2u);
    
    int maxK = mxGetScalar(input_maxK);
    int Kest=maxK;
    double alpha_W[D][4]; 
    //double maxR = mxGetScalar(input_maxR);
    int Nsim = mxGetScalar(input_Nsim);
    
    int C[D];
    int R[D];
    //double W[D][4];
    int maxR = 0;
    for (int d=0; d<D; d++){
         C[d]=(int)C_dou[d];
         R[d]=(int)R_dou[d];
         //printf("%d ",R[d]);
         if (R[d]>maxR){
            maxR=R[d];
         }
         for (int i=0; i<4; i++){
            alpha_W[d][i]=W_dou[D*i+d];
         }
//          for (int i=0; i<4; i++){
//             W[d][i]=W_dou[D*i+d];
//          }
    }
   
    
    gsl_matrix_view Xview = gsl_matrix_view_array(X_dou, D,N);
    gsl_matrix *X = &Xview.matrix;
    
    gsl_matrix_view XTview = gsl_matrix_view_array(XT_dou, D,N);
    gsl_matrix *XT = &XTview.matrix;
 
    
    //...............BODY CODE.......................//
    
    //.....INIZIALIZATION........//
    //double s2u=1;
    double su= sqrt(s2u);
    double sY= sqrt(s2Y);
    double sYd=sY;
    double s2uy= (s2Y+s2u);
    double suy= sqrt(s2Y+s2u);
    double Wint=2;//0.5; 
    //double s2theta=1;
    // random numbers
    srand48(time(NULL));
    //srand48(0);

    gsl_rng *seed = gsl_rng_alloc(gsl_rng_taus);
    time_t clck=time(NULL);
    gsl_rng_set(seed, clck);
//     gsl_rng_set(seed, 0);
    
    // auxiliary variables
    double W[D][4];
    double maxX[D], minX[D], meanX[D], countX, sumX;
    double theta_L[D], theta_H[D], theta_dir[D];
    int S[N][D];
    for (int d =0; d<D; d++){
       double p[4];
       gsl_ran_dirichlet (seed, 4, alpha_W[d], p);
       //printf("W=%f , %f, %f \n \n",  p[0],  p[1],  p[2]);
       for (int i=0;i<4; i++){
            W[d][i]=p[i];
        }
//        printf("W=%f , %f, %f, %f \n \n",  p[0],  p[1],  p[2], p[3]);
        sumX=0;
        countX=0;
        maxX[d]=gsl_matrix_get (X, d, 0);
        minX[d]=gsl_matrix_get (X, d, 0);
        for (int n=0; n<N; n++){
            S[n][d]=0;
            double xnd=gsl_matrix_get (X, d, n);
            if (xnd!=-1 && xnd>maxX[d]){
                maxX[d]=xnd;
            }
            if (xnd!=-1 && xnd<minX[d]){
                minX[d]=xnd;
            }
            if (xnd!=-1 || !gsl_isnan(xnd)){
                sumX=sumX+xnd;
                countX++;
            }
        }
        if (C[d]==1 || C[d]==2){
            meanX[d]=sumX/countX;
            double epsilon=  (maxX[d] -  minX[d])/10000;
            theta_dir[d]=fre_1(GSL_MAX_DBL(fabs(minX[d]), fabs(maxX[d]))+epsilon, 2/maxX[d],0);
            //theta_dir[d]= GSL_MAX_DBL(fabs(minX[d]), fabs(maxX[d]))+epsilon; 
            theta_L[d]= minX[d]-epsilon;
            theta_H[d]= maxX[d]+epsilon;
//             printf("meanX = %f \n",meanX[d]);
//             printf("epsilon = %f \n",epsilon);
//             printf("theta_dir = %f \n",theta_dir[d]);
//             printf("theta_L = %f and theta_H = %f \n",theta_L[d], theta_H[d]);
        }
    }
    
        
    gsl_matrix *Z= gsl_matrix_calloc(Kest,N);
    gsl_matrix **Yreal=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Ypos=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Yint=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Ydir=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    gsl_matrix **Ybin=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    gsl_matrix **Ycat=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Yord=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Ycount=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    
    gsl_matrix **Breal=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Bpos=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Bint=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Bdir=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    gsl_matrix **Bbin=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    gsl_matrix **Bcat=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Bord=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Bcount=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_vector **theta=(gsl_vector **) calloc(D,sizeof(gsl_vector*));
    gsl_matrix *EYE= gsl_matrix_alloc(Kest,Kest);
    gsl_matrix_set_identity (EYE);
    
//     gsl_matrix **Zreal=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Zpos=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Zint=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Zdir=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     
//     gsl_matrix **Zcat=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Zord=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
//     gsl_matrix **Zcount=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    
    for (int d =0; d<D; d++){
        
        gsl_vector_view Bd_view;
        gsl_vector *muB = gsl_vector_calloc(Kest);
        gsl_matrix *SB= gsl_matrix_alloc(Kest,Kest);
        gsl_matrix_set_identity (SB);
        gsl_matrix_scale (SB, s2B);
        //Initialize Y 
        
        if (C[d]==1 || C[d]==2){ //Continuous all         
            double xnd;
            int snd;
//             Y[d] =gsl_matrix_calloc(R[d],N);
            Yreal[d] = gsl_matrix_calloc(1,N);
           
            Breal[d] = gsl_matrix_alloc(Kest,1);
            Bd_view =  gsl_matrix_subcolumn (Breal[d], 0, 0, Kest);
            mvnrnd(&Bd_view.vector, SB, muB, Kest, seed);
            
            double p[4];
            for (int i=0;i<=4; i++){
                p[i]=W[d][i];
            }
            
            for (int n=0; n<N; n++){
                snd=mnrnd(p, 4)+1;
                S[n][d]=snd;
                xnd=gsl_matrix_get (X, d, n);
                //Real
                if (xnd==-1 || gsl_isnan(xnd)){
                    gsl_matrix_set (Yreal[d], 0, n, gsl_ran_gaussian (seed, sY));
                }else{
                    gsl_matrix_set(Yreal[d], 0, n, fre_1(xnd,2/(maxX[d]-meanX[d]),meanX[d]));
                }
            }   
        }
        else if (C[d]==3){ //Binary 
            int xnd;
            Ybin[d] =gsl_matrix_calloc(1,N);
            Bbin[d] = gsl_matrix_alloc(Kest,1);
            Bd_view =  gsl_matrix_subcolumn (Bbin[d], 0, 0, Kest);
            mvnrnd(&Bd_view.vector, SB, muB, Kest, seed);
           
            for (int n=0; n<N; n++){
                xnd=(int)gsl_matrix_get (X, d, n);
                if (xnd==-1 || gsl_isnan(xnd)){
                    gsl_matrix_set (Ybin[d], 0, n, gsl_ran_gaussian (seed, sY));
                }else if (xnd==1){
                    gsl_matrix_set(Ybin[d], 0, n, truncnormrnd(0, sY, GSL_NEGINF, 0));
                }else if (xnd==2){
                    gsl_matrix_set(Ybin[d], 0, n,truncnormrnd(0, sY, 0, GSL_POSINF));
                }
            }
        }
        else if (C[d]==4){ //Discrete
            int xnd, snd;
//             Y[d] =gsl_matrix_calloc(R[d],N);
            Ycat[d] = gsl_matrix_calloc(R[d],N);
           
            Bcat[d] = gsl_matrix_alloc(Kest,R[d]);
                      
            double p[3];
            for (int i=0;i<=3; i++){
                p[i]=W[d][i];
            }
            
            Bd_view =  gsl_matrix_subcolumn (Bcat[d], 0, 0, Kest);
            gsl_vector_set_all (&Bd_view.vector,-1);
            for(int r=1; r<R[d]; r++){
                Bd_view =  gsl_matrix_subcolumn (Bcat[d], r, 0, Kest);
                mvnrnd(&Bd_view.vector, SB, muB, Kest, seed);    
            }

            for (int n=0; n<N; n++){
                xnd=(int)gsl_matrix_get (X, d, n);
                //Categorical       
                if (xnd==-1 || gsl_isnan(xnd)){
                    for(int r=0; r<R[d]; r++){
                        gsl_matrix_set (Ycat[d], r, n, gsl_ran_gaussian(seed, sYd));
                    }
                }else{
                    gsl_matrix_set (Ycat[d], xnd-1, n, truncnormrnd(0, sYd, 0, GSL_POSINF));
                    for(int r=0; r<R[d]; r++){
                        if (r!=xnd-1){
                            gsl_matrix_set (Ycat[d], r, n, truncnormrnd(0, sYd, GSL_NEGINF, gsl_matrix_get (Ycat[d], xnd-1, n)));
                        }
                    }
                }              
            }
        }
        gsl_vector_free(muB);
        gsl_matrix_free(SB);
           
    }
    
    gsl_matrix **Preal=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    gsl_matrix **Pbin=(gsl_matrix **) calloc(D,sizeof(gsl_matrix*));
    gsl_matrix **Pcat=(gsl_matrix **) calloc(D*maxR,sizeof(gsl_matrix*));
    for (int d =0; d<D; d++){
        if (C[d]== 1 || C[d]==2){
            Preal[d]= gsl_matrix_calloc(Kest,Kest);
        }
        else if (C[d]== 3){
            Pbin[d]= gsl_matrix_calloc(Kest,Kest);
        }
        else if (C[d]== 4){
            for(int r=0; r<maxR; r++){
                if (r<R[d]){
                    Pcat[d*maxR+r]= gsl_matrix_calloc(Kest,Kest);
                }
            }
        }
    }
    
    int countErr=0;
    int countErrC=0;
     //....Body functions....//      
    for (int it=0; it<Nsim; it++){
        if (countErr>0 || countErrC>0){break;}
        for (int d =0; d<D; d++){
            if (C[d]== 1 || C[d]== 2){
                gsl_matrix_set_zero (Preal[d]);
                matrix_multiply(Breal[d],Breal[d],Preal[d],1,1,CblasNoTrans,CblasTrans);
            }
            else if (C[d]== 3){
                gsl_matrix_set_zero (Pbin[d]);
                matrix_multiply(Bbin[d],Bbin[d],Pbin[d],1,1,CblasNoTrans,CblasTrans);
            }
            else if (C[d]== 4){
                gsl_matrix_view Bd_view;
                for(int r=0; r<maxR; r++){
                    if (r<R[d]){
                        Bd_view =  gsl_matrix_submatrix (Bcat[d], 0, r, Kest,1);
                        gsl_matrix_set_zero (Pcat[d*maxR+r]);
                        matrix_multiply(&Bd_view.matrix,&Bd_view.matrix,Pcat[d*maxR+r],1,1,CblasNoTrans,CblasTrans);
                    }
                }
            }
       }
        
        // Sampling Z
        gsl_matrix *muZ = gsl_matrix_alloc(Kest,1);
        gsl_matrix *SZ= gsl_matrix_alloc(Kest,Kest);
        gsl_matrix *aux = gsl_matrix_alloc(Kest,1);
        for (int n=0; n<N; n++){
                gsl_matrix_view Bd_view;
                gsl_matrix_set_zero(SZ);
                gsl_matrix_set_zero (muZ);
                gsl_matrix_set_zero (aux);
                for (int d =0; d<D; d++){
                    if (C[d]== 1 || C[d]== 2){
                        gsl_matrix_add (SZ, Preal[d]);
                        gsl_matrix_memcpy (aux, Breal[d]);
                        gsl_matrix_scale (aux, gsl_matrix_get(Yreal[d], 0, n));
                        gsl_matrix_add (muZ, aux);
                    }else if (C[d]== 3){
                        gsl_matrix_add (SZ, Pbin[d]);
                        gsl_matrix_memcpy (aux, Bbin[d]);
                        gsl_matrix_scale (aux, gsl_matrix_get(Ybin[d], 0, n));
                        gsl_matrix_add (muZ, aux);
                    }
                    else if (C[d]== 4){
                        int r= (int)gsl_matrix_get (X, d, n);
                        double auxY= GSL_NEGINF;
                        if (r==-1 || isnan(r)){
                            for(int r2=0; r2<R[d]; r2++){
                                if(gsl_matrix_get (Ycat[d], r2, n)>auxY){r=r2;}
                            }
                        }else{
                            r=r-1;
                            gsl_matrix_add (SZ, Pcat[d*maxR+r]);
                            if (r>0){
                                Bd_view =  gsl_matrix_submatrix (Bcat[d],0,r, Kest,1);
                                gsl_matrix_memcpy (aux, &Bd_view.matrix);
                                gsl_matrix_scale (aux, gsl_matrix_get(Ycat[d], r, n));
                                gsl_matrix_add (muZ, aux);
                            }
                        }
                    }
                }

                matrix_multiply(EYE,EYE,SZ,1/s2Z,1/s2Y,CblasNoTrans,CblasNoTrans);
                inverse(SZ, Kest);
                matrix_multiply(SZ,muZ,aux,1/s2Y,0,CblasNoTrans,CblasNoTrans);
                gsl_vector_view MuB_view =  gsl_matrix_column (aux, 0);
                gsl_vector_view Z_view =  gsl_matrix_column (Z,n);
                mvnrnd(&Z_view.vector, SZ, &MuB_view.vector, Kest, seed);
                  
        }
        gsl_matrix_free(SZ); 
        gsl_matrix_free(muZ);
        gsl_matrix_free(aux);
        
        for (int d =0; d<D; d++){
            // Sampling Y
            gsl_matrix_view Zn;
            gsl_matrix_view Bdm_view;
            gsl_matrix *muy;               
            muy= gsl_matrix_alloc(1,1);
            if (C[d]== 1 || C[d]== 2){  
                double xnd;
                for (int n=0; n<N; n++){
                    xnd=gsl_matrix_get (X, d, n);
                    Zn = gsl_matrix_submatrix (Z, 0, n, Kest, 1);
                    Bdm_view = gsl_matrix_submatrix (Breal[d], 0, 0, Kest, 1);
                    matrix_multiply(&Zn.matrix,&Bdm_view.matrix,muy,1,0,CblasTrans,CblasNoTrans); 
                    if (xnd==-1 || gsl_isnan(xnd)){
                        gsl_matrix_set (Yreal[d], 0, n, gsl_matrix_get(muy,0,0)+  gsl_ran_gaussian (seed, sY));
                    }else{
                        gsl_matrix_set (Yreal[d], 0, n, (fre_1(xnd,2/(maxX[d]-meanX[d]),meanX[d])/s2u + gsl_matrix_get(muy,0,0)/s2Y)/(1/s2Y+1/s2u) +  gsl_ran_gaussian (seed, sqrt(1/(1/s2Y+1/s2u))));
                    }
                }                     
                gsl_matrix_free(muy);
                
            }else if (C[d]== 3){
                int xnd;
                muy= gsl_matrix_alloc(1,1);
                for (int n=0; n<N; n++){
                    xnd=(int)gsl_matrix_get (X, d, n);
                    Zn = gsl_matrix_submatrix (Z, 0, n, Kest, 1);
                    Bdm_view = gsl_matrix_submatrix (Bbin[d], 0, 0, Kest, 1);
                    matrix_multiply(&Zn.matrix,&Bdm_view.matrix,muy,1,0,CblasTrans,CblasNoTrans);
                    if (xnd==-1 || gsl_isnan(xnd)){
                         gsl_matrix_set (Ybin[d], 0, n, gsl_matrix_get(muy,0,0)+  gsl_ran_gaussian (seed, sY));
                    }else if (xnd==1){
                        gsl_matrix_set (Ybin[d], 0, n, truncnormrnd(gsl_matrix_get(muy,0,0), sY, GSL_NEGINF, 0));
                    }else if (xnd==2){
                        gsl_matrix_set (Ybin[d], 0, n, truncnormrnd(gsl_matrix_get(muy,0,0), sY, 0, GSL_POSINF));
                    }
                }
                gsl_matrix_free(muy);
                
            }else if (C[d]== 4){
                int xnd;
                gsl_matrix *muyCat= gsl_matrix_alloc(1,R[d]);
                for (int n=0; n<N; n++){                   
                    xnd=(int)gsl_matrix_get (X, d, n);   
                    Zn = gsl_matrix_submatrix (Z, 0, n, Kest, 1);
                    matrix_multiply(&Zn.matrix,Bcat[d],muyCat,1,0,CblasTrans,CblasNoTrans);
                    if (xnd==-1 || gsl_isnan(xnd)){
                        for(int r=0; r<R[d]; r++){
                            gsl_matrix_set (Ycat[d], r, n, gsl_matrix_get(muyCat,0,r)+gsl_ran_gaussian (seed, sYd));
                        }
                    }else{                           
                        gsl_matrix_set(Ycat[d], xnd-1, n, truncnormrnd(gsl_matrix_get(muyCat,0,xnd-1), sYd, 0, GSL_POSINF)); 
                        if(gsl_isinf(gsl_matrix_get (Ycat[d], xnd-1, n))){
                            printf("n=%d, d=%d,xnd=%d, muy=%f, y=%f \n",n,d,xnd-1, gsl_matrix_get(muyCat,0,xnd-1), gsl_matrix_get (Ycat[d], xnd-1, n));}
                        for(int r=0; r<R[d]; r++){
                            if (r!=xnd-1){
                                gsl_matrix_set (Ycat[d], r, n, truncnormrnd(gsl_matrix_get(muyCat,0,r), sYd, GSL_NEGINF, gsl_matrix_get (Ycat[d], xnd-1, n)));
                            }
                        }
                    } 
                } 
                gsl_matrix_free(muyCat);
            }
            
            
             // Sampling B
            gsl_matrix *muB = gsl_matrix_alloc(Kest,1);
            gsl_matrix *SB= gsl_matrix_alloc(Kest,Kest);
            gsl_matrix *aux = gsl_matrix_alloc(Kest,1);
            gsl_matrix_set_identity (SB);
            gsl_matrix_view Yd_view;
            gsl_vector_view MuB_view, Bd_view;
            
            if (C[d]== 1 || C[d]== 2){ //Continuous all
                 matrix_multiply(Z,Z,SB,1/s2Y,1/s2B,CblasNoTrans,CblasTrans);
                 inverse(SB, Kest);
                 
                 matrix_multiply(Z,Yreal[d],muB,1/s2Y,0,CblasNoTrans,CblasTrans);
                 matrix_multiply(SB,muB,aux,1,0,CblasNoTrans,CblasNoTrans);
                 gsl_vector_view MuB_view =  gsl_matrix_column (aux, 0);
                 gsl_vector_view Bd_view =  gsl_matrix_column (Breal[d],0);
                 mvnrnd(&Bd_view.vector, SB, &MuB_view.vector, Kest, seed);   
           
            }else if (C[d]== 3){ //Binary
                 matrix_multiply(Z,Z,SB,1/s2Y,1/s2B,CblasNoTrans,CblasTrans);
                 inverse(SB, Kest);
                 matrix_multiply(Z,Ybin[d],muB,1/s2Y,0,CblasNoTrans,CblasTrans);
                 matrix_multiply(SB,muB,aux,1,0,CblasNoTrans,CblasNoTrans);
                 gsl_vector_view MuB_view =  gsl_matrix_column (aux, 0);
                 gsl_vector_view Bd_view =  gsl_matrix_column (Bbin[d],0);
                 mvnrnd(&Bd_view.vector, SB, &MuB_view.vector, Kest, seed);   

            }else if (C[d]== 4){//Discrete
                 matrix_multiply(Z,Z,SB,1/s2Y,1/s2B,CblasNoTrans,CblasTrans);
                 inverse(SB, Kest);
                 
                 for(int r=1; r<R[d]; r++){
                     Yd_view =  gsl_matrix_submatrix (Ycat[d],r,0, 1,N);
                     matrix_multiply(Z,&Yd_view.matrix,muB,1/s2Y,0,CblasNoTrans,CblasTrans);
                     matrix_multiply(SB,muB,aux,1,0,CblasNoTrans,CblasNoTrans);
                     MuB_view =  gsl_matrix_column (aux, 0);
                     Bd_view = gsl_matrix_column (Bcat[d],r);
                     mvnrnd(&Bd_view.vector, SB, &MuB_view.vector, Kest, seed);  
                 }
            }          
           
            gsl_matrix_free(SB);
            gsl_matrix_free(muB);
            gsl_matrix_free(aux) ;
            
        }       
    }

    
    printf("countErr %d \n",countErr);
    printf("countErrC %d \n",countErrC);
    
    printf("Computing LIKelihood \n");
    
    double LIK[D][N];
    for (int n=0; n<N; n++){ 
        for (int d=0; d<D; d++){
            LIK[d][n]=0;
            double xnd=gsl_matrix_get (X, d, n);
            if(xnd==-1){
                double tnd=gsl_matrix_get (XT, d, n);
                if(tnd!=-1){
                    //printf("xmiss %f \n",tnd);
                    LIK[d][n]=0;
                    if (C[d]== 1 || C[d]== 2){
                        gsl_matrix *aux = gsl_matrix_alloc(1,1);
                        gsl_matrix *Baux = gsl_matrix_alloc(Kest,1);
                        gsl_matrix_view Zn_view = gsl_matrix_submatrix (Z,0,n, Kest,1);
                        gsl_matrix_view Bd_view;

                        Bd_view = gsl_matrix_submatrix (Breal[d],0,0, Kest,1);
                        matrix_multiply(&Zn_view.matrix,&Bd_view.matrix,aux,1,0,CblasTrans,CblasNoTrans);                     
                        LIK[d][n]= xpdf_re(tnd, 2/(maxX[d]-meanX[d]),meanX[d], gsl_matrix_get (aux, 0, 0), s2Y, s2u);
                      
                        gsl_matrix_free(aux);
                        gsl_matrix_free(Baux);
                    }else if (C[d]== 4){
                        gsl_matrix *aux = gsl_matrix_alloc(1,1);
                        gsl_matrix *Baux = gsl_matrix_alloc(Kest,1);
                        gsl_matrix_view Zn_view = gsl_matrix_submatrix (Z,0,n, Kest,1);
                        gsl_matrix_view Bd_view, Bd_view2;
                        int r= (int)gsl_matrix_get (XT, d, n)-1;
                        Bd_view = gsl_matrix_submatrix (Bcat[d],0,r, Kest,1);
                        double sumC=0;
                        double prodC[100];
                        double u[100];
                        for(int ii=0; ii<100; ii++){
                            u[ii]= gsl_ran_gaussian (seed, sYd);
                            prodC[ii]=1;
                        }                            

                        for(int r2=0; r2<R[d]; r2++){
                            if (r2!=r){
                                Bd_view2 = gsl_matrix_submatrix (Bcat[d],0,r2, Kest,1);
                                gsl_matrix_memcpy(Baux,&Bd_view.matrix);
                                gsl_matrix_sub (Baux, &Bd_view2.matrix);  
                                matrix_multiply(&Zn_view.matrix,Baux,aux,1,0,CblasTrans,CblasNoTrans);
                                for(int ii=0; ii<100; ii++){
                                    prodC[ii] =prodC[ii]* gsl_cdf_ugaussian_P (u[ii]+ gsl_matrix_get (aux, 0, 0));
                                }
                            }
                        }
                        for(int ii=0; ii<100; ii++){sumC=sumC+prodC[ii];}
                        LIK[d][n]=logFun(sumC/100);
                           
                        gsl_matrix_free(aux);
                        gsl_matrix_free(Baux);
                    
                    }    
                }
            }
        }
    }
    
    
//...............SET OUTPUT POINTERS.......................//
// #define output_K plhs[0]
    
    output_K = mxCreateDoubleScalar(Kest);
    output_countErr = mxCreateDoubleScalar(countErr+countErrC);
    
 
    output_W = mxCreateDoubleMatrix(D, 4,mxREAL);
    double *pW=mxGetPr(output_W);    
    for (int d=0; d<D; d++){
         for (int i=0; i<4; i++){
             pW[D*i+d]=0;
         }
    }
    
    output_LIK = mxCreateDoubleMatrix(D, N,mxREAL);
    double *pL=mxGetPr(output_LIK);    
    for (int d=0; d<D; d++){
         for (int n=0; n<N; n++){
             pL[D*n+d]=LIK[d][n];
         }
    }

    
    //..... Free memory.....//
    for (int d=0; d<D; d++){    
//         gsl_matrix_free(Y[d]);
        
        gsl_matrix_free(Breal[d]);
       
        gsl_matrix_free(Bbin[d]);
        gsl_matrix_free(Bcat[d]);
        
        gsl_matrix_free(Preal[d]);
        
        gsl_matrix_free(Pbin[d]);
        gsl_matrix_free(Pcat[d]);
           
        gsl_matrix_free(Yreal[d]);
        
        gsl_matrix_free(Ycat[d]);

    }
    gsl_matrix_free(Z);
    gsl_matrix_free(EYE);
    
    free(Breal);
    free(Bbin);
    free(Bcat);


    free(Preal);

    free(Pbin);
    free(Pcat);

    free(Yreal);
    
    free(Ycat);
    
    gsl_rng_free(seed);

}
    
