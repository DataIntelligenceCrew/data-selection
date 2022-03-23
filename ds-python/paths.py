'''
Define all path variables here

We have the following directory structure for storage:


/localdisk3/data-selection/data
    datasets/ : contains all the datasets with the full train and test splits
        cifar10
        mnist
        fashion-mnsit
        svhn : maybe
    metadata/ : contains the feature vectors and the posting lists for various coverage factors and paritions
        $dataset_name/
            feature-vector-file
            labels.txt
            $coverage_threshold/
                $number_partitions/ 
                    list of all posting list files
    
    model-data/ : contains the data to train ML model based on the solution 
                  & stores the model runs and checkpoints, with tensorboard logs
        $dataset_name/
            $distribution_req/
                $coverage_factor/
                    $algo_type/
                        train/ : contains the data with subfolders for each class
                        $model_type
                            tblogs/ : contains the tensorboardX logs
                            checkpoints/ : stores the model checkpoints over epochs
                            models/: final model parameter dict
    
    runs/ : contains all the algo runs and the metric files
        metric_files/ 
        solution_files/

'''
import torch


# define device type, number of GPUs available
DEVICE_IDS = [0, 1, 2, 3]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paths for model-data/
MODEL_OUTPUT_DIR = '/localdisk3/data-selection/data/model-data/{0}/{1}/{2}/{3}/{4}/'
INPUT_IMG_DIR_CORESET = '/localdisk3/data-selection/data/model-data/{0}/{1}/{2}/{3}/train/'
INPUT_IMG_DIR_FULLDATA = '/localdisk3/data-selection/data/datasets/{0}/train/'
ALEXNET_MODEL_ID = 'alexnet_epochs_{0}_lr_{1}_bs_{2}_seed{3}'
CONVNET_MODEL_ID = 'convnet_W_{0}_D_{1}_N_{2}_A_{3}_P_{4}_epochs_{5}_lr_{6}_bs_{7}_seed{8}'
TEST_IMG_DIR = '/localdisk3/data-selection/data/datasets/{0}/test/'



# paths for runs/
# (dataset_name, coverage_factor, distribution_req, algo_type)
METRIC_FILE = '/localdisk3/data-selection/data/runs/metric_files/{0}_{1}_{2}_{3}.txt'
SOLUTION_FILENAME = '/localdisk3/data-selection/data/runs/solution_files/{0}_{1}_{2}_{3}.txt'

# paths for metadata/
FEATURE_VECTOR_LOC = '/localdisk3/data-selection/data/metadata/{0}/vectors-alexnet'
POSTING_LIST_LOC = '/localdisk3/data-selection/data/metadata/{0}/{1}/{2}/'
LABELS_FILE_LOC = '/localdisk3/data-selection/data/metadata/{0}/labels.txt'



