# Timings

On intel core i7 2.80Ghz

```
$ python3 train.py -x_train_dir datasets/train-images.idx3-ubyte -y_train_dir datasets/train-labels.idx1-ubyte
Reading dataset...Done in 0.0375 s
Preparing dataset...Done in 15.3710 s
Training...Done in 0.0359 s
Validate model...Done in 122.8647 s
```

```
$ python3 predict.py -x_test_dir datasets/t10k-images.idx3-ubyte -y_test_dir datasets/t10k-labels.idx1-ubyte 
Reading dataset...Done in 0.0057 s
Preparing dataset...Done in 2.0897 s
Reading model...Done in 0.0065 s
Validate model...Done in 21.3732 s
```

