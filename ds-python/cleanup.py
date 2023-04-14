import os 
import shutil






main_dir = "/localdisk3/data-selection/data/model-data/{0}/{1}/{2}/{3}/convnet/{4}"

datasets = ['cifar10', 'cifar100', 'mnist', 'fasion-mnist']

DR = ['0', '50', '100', '200', '300', '400', '500', '600', '700', '800', '900']

CF = ['0', '10', '15', '20', '25', '30', '35', '40', '45', '50']

algos = ['greedyNC', 'greedyC_random', 'greedyC_group', 'MAB', 'k_centersNC']


folders = ['tblogs', 'checkpoints']


for ds in datasets:
    for dr in DR:
        for cf in CF:
            for alg in algos:
                for f in folders:
                    folder_path = main_dir.format(ds, dr, cf, alg, f)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)