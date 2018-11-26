## Report for Exam 2
### By Xiaochi (George) Li

*All the "we" mentioned in the report is just a idiom in academic writing, this report is completed by myself*

## Question 1-5
Just follow the instructions.  
There are two possible bugs:

1. When download the files from Github repo and upload it to our VM, there may be encoding issue.  
Fix: Copy the file from the repo 
2. In step 3, the path of BUILD should also be changed   
Fix: Change it to the caffe root on this VM

## Question 6
The loss decrease with the number of iteration and the Test Accuracy increase with the number of accuracy.   
It means the training is effective.

![](./loss.png)

![](./accuracy.png)

## Question 7
We modified the original ```train_minst.py ``` to let the program visualize the kernels in convolution layer 2.   
The kernels looks like parts of a number, like a vertical line ,part of a circle, or a corner.  
These may be helpful in identifying the hand writing numbers.  

![](./kernel_conv1.png)

![](./kernel_conv2.png)

## Question 8
The performance of Convolution network is much better that the multilayer networks in Exam 1

## Question 9

|Mini batch size|Average Time|Total Time 20k epoch)|
|----|----|----|
|32|0.00290|58.04|
|64|0.00346|69.21|
|128|0.00463|92.61|
|256|0.00745|149.15|
|512|0.0128|257.7|

After making the size of mini batch 512, the average time per epoch slowed down by 4.44 times.

|Batch size = 32|Batch size = 512|
|----|----|
|![](./batch32.png)|![](./batch512.png)|

We can see that the advantage of a larger batch size is that the fluctuation of the loss function is smaller.
Which means the algorithm converges in a smaller epoch. However the disadvantage is that it will cost more time.

## Question 10
Add the dropout layer in ```lenet_train_test.prototxt```:
```text
layer {
  name: "drop"
  type: "Dropout"
  bottom: "ip1"
  top: "ip1"
  dropout_param {
    dropout_ratio: 0.5
  }
}
```
The accuracy of test after using drop out is still 1.00, so there is no difference.

|No dropout|Dropout=0.5|
|----|----|
|![](./batch32.png)|![](./drop50.png)|

## Question 11

Original:  

|Layer|Parameter|Data size after this layer|Number of weight|
|----|----|----|----|
|Conv1|20x5x5,s=1|24x24x20|20x5x5|
|Pool1|2x2,s=2|12x12x20|0|
|Conv2|50x5x5,s=1|8x8x20x50|50x5x5|
|Pool2|2x2,s=2|4x4x20x50=16000|0|
|ip1|-|500|16000x500|
|ip2|-|10|500x10|
|**Sum**|-|-|8006750|

New structure: Change kernel size to 3x3 and add one Convolution layer.

|Layer|Parameter|Data size after this layer|Number of weight|
|----|----|----|----|
|Conv1|10x3x3,s=1|26x26x10|10x3x3|
|Pool1|2x2,s=2|13x13x10|0|
|Conv2|10x3x3,s=1|11x11x10x10|10x3x3|
|Pool2|2x2,s=2|6x6x10x10|0|
|Conv3|10x3x3|4x4x10x10x10=16000|10x3x3|
|ip1|-|500|16000x500|
|ip2|-|10|500x10|
|**Sum**|-|-|8005270|

|Original|New Structure|
|----|----|
|![](./q11-a.JPG)|![](./q11-b.JPG)|

|Structure|Loss|Accuracy|
|----|----|----|
|Original|![](./SGD-loss.png)|![](SGD-accuracy.png)|
|New|![](new-loss.png)|![](new-accuracy.png)|

|Structure|Average Time|Total Time 20k epoch)|
|----|----|----|
|Original|0.00290|58.04|
|New|0.002168|43.37|

Finding: The new structure with smaller kernel size trains faster, however the convergence is slower.

 
## Question 12
Modify to [Adam optimizer](http://caffe.berkeleyvision.org/tutorial/solver.html) in ```train_mnist.py```

```python
solver = caffe.AdamSolver('lenet_solver.prototxt')
```


|Optimizer|Loss|Accuracy|
|----|----|----|
|SGD|![](./SGD-loss.png)|![](./SGD-accuracy.png)|
|Adam|![](./adam-loss.png)|![](./adam-accuracy.png)|

Finding: SGD works much better than Adam.
