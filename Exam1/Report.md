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

## Question 5
We copied Q4a to Q5a as the first optimizer option. And tried several optimizer as the table shows.   
We used learning_rate = 0.1, momentum = 0.9 as hyper-parameter.

|Name|Accuracy|Average time per epoch|Optimizer|
|----|----|----|----|
|Q5a|45%|8.38|optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)|
|Q5b|37%|8.31|optim.SGD(net.parameters(), lr=learning_rate)|
|Q5c|21%|8.25|optim.RMSprop(net.parameters(), lr=learning_rate)|
|Q5d|19%|8.28|optim.RMSprop(net.parameters(), lr=learning_rate, momentum=momentum)|
|Q5e|19%|8.40|optim.Adam(net.parameters(),lr=learning_rate)|

Conclusion: Stochastic Gradient Descent with momentum is the best optimizer we have. 

## Question 6
We copied Q4a to Q6a as the first transfer function option.  

|Name|Transfer function|Accuracy|Average time per epoch|
|----|----|----|----|
|Q6a|ReLU|45%|8.46|
|Q6b|Sigmoid|36%|8.34|
|Q6c|Leaky ReLU|45%|8.45|
|Q6d|PReLU|45%|8.35|
|Q6e|Tanh|42%|8.29|

Conclusion: ReLU based transfer function works better than Sigmoid and Tanh as expected.  
The reason is that ReLU preserve a bigger gradient when the input is far from zero. Thus prevented gradient shrinkage.



