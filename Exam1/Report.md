## Report for Exam 1
### By Xiaochi (George) Li

*All the "we" mentioned in the report is just a idiom in academic writing, this report is completed by myself *

## Question 1
After fixing all the bugs and errors. The overall accuracy of the Neural Network is around 44%.   
Setting: Epoch=10, learning rate=0.1,momentum = 0.9, hidden size=30.  
We will fine tune the hyper-parameters in the future.  

## Question 2
Time for running on CPU is 8.5 sec per epoch, however using cuda didn't speed up the training significantly(reduce 0.3 sec
per epoch).  
Overall accuracy is 44% with the same hyper-parameter.

## Question 3 
The program is set up to perform mini-batch gradient descent. Because during each iteration of the training,
the program takes an input which has shape batch_size * input_size. So we can know that it's a mini-batch gradient descent.

## Question 4
We modified the Neural Network we constructed in Q2, and used one hidden layer with 60 neurons.   
The structure is 3072-60-10,and the total number of parameters is 60x3072+60x1 + 10x60+10x1 = 184,990  
In Q4b, we tried to make the neural network deeper to 3 hidden layers, the structure is 3072-59-40-20-10, the total number of 
parameter is 184737.

|Name|Structure|Accuracy|Average time per epoch|
|----|----|----|----|
|Q4a|3072-60-10|45%|8.722|
|Q4b|3072-59-40-20-10|23%|8.557|

It can be easily seen that more layers will harm the performance.



