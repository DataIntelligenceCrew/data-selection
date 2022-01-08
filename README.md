# Data-Selection

## Things to keep in mind
- Always look at the open issues
- Make sure to add comments before pushing
- Use variable name that make sense
- Open Pull Requests everytime you push 

## Plan for Week (1/8 - 1/15):

- **ResNet and FAISS** : convert the image dataset, into vectors and build the FAISS Index over that
    
        - (Jiwon & Rosie)
        - Dataset (CIFAR-10) : https://www.cs.toronto.edu/~kriz/cifar.html
        - Download the dataset to /localdisk3/ on node 2x16a
        - Go through ResNet Code :  https://github.com/KaimingHe/deep-residual-networks
        - Download the code and convert the CIFAR-10 images to vector embeddings
        - Load the vector embeddings into a FAISS Index build on the GPU
        - FAISS Index code can be found here : https://github.com/DataIntelligenceCrew/koios-semantic-search

- **Algorithm and Complexity Analysis** : pseudocode for the algorithm and complexity analysis
    - (Pranay)
    - Write out the algorithm 