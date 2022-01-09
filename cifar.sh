# Downloads CIFAR 10 dataset and unzips all content to "../cifar" folder
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o cifar.tar.gz
tar -xzf cifar.tar.gz
rm cifar.tar.gz
mkdir cifar
mv cifar-10-batches-py/* cifar/
rm -rf cifar-10-batches-py