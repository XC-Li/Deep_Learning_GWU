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
