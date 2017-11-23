

#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "kernels.h"
#include "LIBIRWLS-predict.h"


/**
 * @brief It the main function to build the executable file to make predictions 
 * on a dataset using a model previously trained using PIRWLS-train or PSIRWLS-train.
 */

int main(int argc, char** argv)
{

    srand(time(NULL));
    
    int MASTER = 0;

    int ierr, num_procs, my_id;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    predictProperties props = parsePredictParameters(&argc, &argv);

    int error = 0;  

    if( my_id == MASTER ) {
        if (argc != 4) {
            error = 4;
        }
    }

   MPI_Bcast(&error, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

   if (error != 0) {
       if (my_id == MASTER) {
           printPredictInstructions();
       }
       MPI_Finalize();
       exit(0);
   } 

    // The name of the files
    char * data_file = argv[1];
    char * data_model = argv[2];
    char * output_file = argv[3];

    if (error != 0) {
        printf("\nRunning with parameters:\n");
        printf("------------------------\n");
        printf("Dataset: %s\n",data_file);
        printf("The model to use: %s\n",data_model);
        printf("The result will be saved in: %s\n",output_file);
        printf("flag l = %d (l = 1 for labeled datasets, l = 0 for unlabeled datasets)\n",props.Labels);
        printf("------------------------\n");
        printf("\n");
    }
  
    model  mymodel;
    
    // Reading the trained model from the file
    if(my_id==0 ) printf("\nReading trained model from file:%s\n",data_model);

    FILE *In = fopen(data_model, "rb");
    if (In == NULL) {
        fprintf(stderr, "Input file with the trained model not found: %s\n",data_model);
        exit(2);
    }
    readModel(&mymodel, In);
    fclose(In);
    if(my_id==0 ) printf("Model Loaded, it contains %d Support Vectors\n\n", mymodel.nSVs);


    // Loading dataset
    if(my_id==0 ) printf("Reading dataset from file:%s\n",data_file);
    MPI_Barrier(MPI_COMM_WORLD);

    svm_dataset dataset;
    In = fopen(data_file, "rb");
    if (In == NULL) {
        fprintf(stderr, "Input file with the training set not found: %s\n",data_file);
        exit(2);
    }
    fclose(In);	  
    if(props.Labels==0){
        dataset=readUnlabeledFile(data_file,my_id, num_procs);
    }else{
        dataset=readTrainFile(data_file, my_id, num_procs);			
    }
    printf("Dataset Loaded, in partition %d it contains %d samples\n", my_id, dataset.l);
    MPI_Barrier(MPI_COMM_WORLD);

    // Set the number of openmp threads
    omp_set_num_threads(props.Threads);

    //Making predictions
    if(my_id==0 ) printf("Classifying data...\n");
    double *predictions;
    if (props.Soft==0){
        predictions=test(dataset,mymodel,props,my_id, num_procs);
    }else{
        predictions=softTest(dataset,mymodel,props, my_id, num_procs);
    }

    if(my_id==0 ) printf("\nWriting output in file: %s \n\n",output_file);
    writeOutput (output_file, predictions,dataset.l,my_id,num_procs);

    freeDataset(dataset);
    freeModel(mymodel);
    free(predictions);

    ierr = MPI_Finalize();
    exit(0);
    return 0;   
}
