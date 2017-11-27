/*
 ============================================================================
 Author      : Roberto Diaz Morales
 ============================================================================
 
 Copyright (c) 2016 Roberto Díaz Morales

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
 (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 ============================================================================
 */

/**
 * @brief Implementation of the training functions of the PSIRWLS algorithm.
 *
 * See PSIRWLS-train.h for a detailed description of its functions and parameters.
 * 
 * For a detailed description of the algorithm and its parameters read the following paper:
 *
 * Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.
 *
 * @file PSIRWLS-train.c
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @see PSIRWLS-train.h
 * 
 */

#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "IOStructures.h"
#include "LIBIRWLS-predict.h"

#include "PSIRWLS-train.h"
#include "kernels.h"
#include "ParallelAlgorithms.h"


/**
 * @cond
 */


int* centroidsxprocess(int id_proc, int tot_proc, int size){

    int* numxproc = calloc(tot_proc, sizeof(int));
    int i;

    if (id_proc==0){
        for (i = 0;i<size;i++) numxproc[rand() % tot_proc]++;
    }

    MPI_Bcast(numxproc, tot_proc, MPI_INT, 0, MPI_COMM_WORLD);

    return numxproc;

} 

int *selectLocalCentroids(int size, svm_dataset dataset){

    int* permut = malloc(dataset.l * sizeof(int));
    int i,j;

    // initial range of numbers
    for(i=0;i<dataset.l;++i){
        permut[i]=i;
    }
    
    for (i = dataset.l-1; i >= 0; --i){
        //generate a random number [0, n-1]
        int j = rand() % (i+1);
        //swap the last element with element at random index
        int temp = permut[i];
        permut[i] = permut[j];
        permut[j] = temp;
    }

    int* centroids = malloc(size * sizeof(int));
    for (i = 0; i < size; i++){
        centroids[i]=permut[i];
    }

    free(permut);
    return centroids;

}


model selectRandomCentroids(svm_dataset dataset,properties props, int id_proc, int tot_proc){

    int* numxproc = centroidsxprocess(id_proc, tot_proc, props.size);
    int i;
    int* centroids = selectLocalCentroids(numxproc[id_proc], dataset);
    int nElem=0;
    svm_sample *iteratorSample;
    svm_sample *classifierSample;

    for (i =0;i<numxproc[id_proc];i++){
        iteratorSample = dataset.x[centroids[i]];
        while (iteratorSample->index != -1){
        	  ++iteratorSample;
            ++nElem;
        }
        ++nElem;
    }
    
    svm_sample* features = (svm_sample *) calloc(nElem,sizeof(svm_sample));
    
    classifierSample = features;

    for (i =0;i<numxproc[id_proc];i++){
        iteratorSample = dataset.x[centroids[i]];
        while (iteratorSample->index != -1){
            classifierSample->index = iteratorSample->index;
            classifierSample->value = iteratorSample->value;
            ++classifierSample;
            ++iteratorSample;
        }
        classifierSample->index = iteratorSample->index;
        ++classifierSample;    
    }
    
    int* elemxproc = calloc(tot_proc, sizeof(int));
    MPI_Allgather(&nElem, 1, MPI_INT, elemxproc, 1, MPI_INT, MPI_COMM_WORLD);
    int* displs = calloc(tot_proc, sizeof(int));
    int totalElements = elemxproc[0];
    for (i=1;i<tot_proc;i++){
        displs[i]=totalElements;
        totalElements=totalElements+elemxproc[i];
    }


    svm_sample* collectedFeatures = calloc(totalElements, sizeof(svm_sample));

    MPI_Datatype MPI_SVM_SAMPLE = getMPI_SVM_SAMPLE();
    MPI_Allgatherv(features,nElem,MPI_SVM_SAMPLE,collectedFeatures,elemxproc,displs,MPI_SVM_SAMPLE, MPI_COMM_WORLD);

    model classifier;
    classifier.Kgamma = props.Kgamma;
    classifier.nSVs = props.size;
    classifier.bias=0.0;
    classifier.kernelType = props.kernelType;
    classifier.nElem = totalElements;

    classifier.weights = (double *) calloc(props.size,sizeof(double));
    classifier.quadratic_value = (double *) calloc(props.size,sizeof(double));
    classifier.x = (svm_sample **) calloc(props.size,sizeof(svm_sample *));
    classifier.features = collectedFeatures;
    
    classifier.x[0]=&(classifier.features[0]);    
    int iterSV=1;
    int aux;
    for(aux=0;aux<(classifier.nElem);aux++){
        if (classifier.features[aux].index == -1){
            if(iterSV<classifier.nSVs) classifier.x[iterSV]=&(classifier.features[aux+1]);
            ++iterSV;
        }else{
            classifier.quadratic_value[iterSV-1]+=classifier.features[aux].value;
        }
    }

    free(numxproc);
    free(elemxproc);
    free(displs);
    free(centroids);
    free(features);
    return classifier;

   
}


/**
 * @brief Is the main function to build the executable file to train a SVM using the PSIRWLS procedure.
 */
  
int main(int argc, char** argv)
{

    srand(time(NULL));
    srand48(time(NULL));

    int MASTER = 0;

    int ierr, num_procs, my_id;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    properties props = parseTrainParameters(&argc, &argv);

    int error = 0;  

    if( my_id == MASTER ) {
        if (argc != 3) {
            error = 4;
        }
    }

   MPI_Bcast(&error, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

   if (error != 0) {
       if (my_id == MASTER) {
           printPSIRWLSInstructions();
       }
       MPI_Finalize();
       exit(0);
   } 


    char * data_file = argv[1];
    char * data_model = argv[2];


   if (error != 0) {
    printf("\nRunning with parameters:\n");
    printf("------------------------\n");
    printf("Training set: %s\n",data_file);
    printf("The model will be saved in: %s\n",data_model);
    printf("Cost c = %f\n",props.C);
    printf("Semiparametric size = %d\n",props.size);


    if(props.kernelType == 0){
        printf("Using linear kernel\n");
    }else{
        printf("Using gaussian kernel with gamma = %f\n",props.Kgamma);
    }
    printf("------------------------\n");
    printf("\n");  
    // Loading dataset
    printf("\nReading dataset from file:%s\n",data_file);
   }


    FILE *In = fopen(data_file, "r");
    if (In == NULL) {
        fprintf(stderr, "Input file with the training set not found: %s\n",data_file);
        exit(2);
    }

    fclose(In);
    svm_dataset dataset = readTrainFile(data_file,my_id,num_procs);
    printf("Dataset Loaded in node %d of %d\nTraining samples: %d\n\n",my_id,num_procs,dataset.l);


    #ifdef OSX    
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    printf("running osx\n");
    #endif

    struct timeval tiempo1, tiempo2;

    omp_set_num_threads(props.Threads);

    printf("Selecting centroids\n");
    gettimeofday(&tiempo1, NULL);

    model modelo=selectRandomCentroids(dataset,props,my_id,num_procs);

    if(my_id==0) printf("Centroids Selected\n");

    omp_set_num_threads(props.Threads);

    initMemory(props.Threads,props.size);

    double * W = IRWLSpar(dataset,modelo,props,my_id,num_procs);

    gettimeofday(&tiempo2, NULL);
    if(my_id==0) printf("Weights calculated in %ld miliseconds\n\n",((tiempo2.tv_sec-tiempo1.tv_sec)*1000+(tiempo2.tv_usec-tiempo1.tv_usec)/1000));

    modelo.weights=W;

    //model modelo = calculatePSIRWLSModel(props, dataset,centroids, W);
	
    freeMemory(props.Threads);
	
    if(my_id==0) printf("Saving model in file: %s\n\n",data_model);	
 
    if(my_id==0){
        FILE *Out = fopen(data_model, "wb");
        storeModel(&modelo, Out);
        fclose(Out);	
    }
	
    freeModel(modelo);
    freeDataset(dataset);
    //free(W);

    ierr = MPI_Finalize();
    exit(0);

}

/**
 * @endcond
 */
