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
                    $groupbased/
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


# define img2vec model and the output dimensions of the fvs
MODELS = {'resnet' : 512, 'alexnet' : 4096, 'densenet' : 1024, 'efficientnet_b0' : 1280, 'efficientnet_b2' : 1408,
            'efficientnet_b3' : 1536, 'efficientnet_b4' : 1792, 'efficientnet_b5' : 2048, 'efficientnet_b6' : 2304, 'efficientnet_b7' : 2560}

# define device type, number of GPUs available
DEVICE_IDS = [0, 1]
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
METRIC_FILE = '/localdisk3/data-selection/data/runs/metric_files/{0}_{1}_{2}_{3}_{4}.txt'
METRIC_FILE2 = '/localdisk3/data-selection/data/runs/metric_files/{0}_{1}_{2}_{3}.txt'
SOLUTION_FILENAME = '/localdisk3/data-selection/data/runs/solution_files/{0}_{1}_{2}_{3}_{4}.txt'
SOLUTION_FILENAME2 = '/localdisk3/data-selection/data/runs/solution_files/{0}_{1}_{2}_{3}.txt'
METRIC_FILE_GROUP = '/localdisk3/data-selection/data/runs/metric_files/{0}_{1}_{2}_{3}_{4}.txt'
SOLUTION_FILENAME_GROUP = '/localdisk3/data-selection/data/runs/solution_files/{0}_{1}_{2}_{3}_{4}.txt'

# paths for metadata/
FEATURE_VECTOR_LOC = '/localdisk3/data-selection/data/metadata/{0}/vectors-{1}'
POSTING_LIST_LOC = '/localdisk3/data-selection/data/metadata/{0}/{1}/{2}/'
POSTING_LIST_LOC_GROUP = '/localdisk3/data-selection/data/metadata/{0}/{1}/{2}/{3}/'
LABELS_FILE_LOC = '/localdisk3/data-selection/data/metadata/{0}/labels.txt'



