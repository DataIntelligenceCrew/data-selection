import alexnet
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_weight', type=float, required=True, default=0.1)
    parser.add_argument('--coreset', type=int, required=True)
    parser.add_argument('--train', type=int, required=True)
    parser.add_argument('--composable', type=int, required=True)
    parser.add_argument('--coverage_factor', type=int, required=True)
    parser.add_argument('--distribution_req', type=int, required=True)
    args = parser.parse_args()
    alexnet.main(args)