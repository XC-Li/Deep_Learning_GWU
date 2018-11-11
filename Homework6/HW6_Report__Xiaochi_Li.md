## Report for homework 6
Author: Xiaochi Li

## Part 1 Pytorch Basics
### E1Q1:  
*Rewrite the "2_tensor_pytorch.py" file in Neural Network Design Book,
Chapter 11 notation (Check the Summary Page) and find out
what are the differences. Explain your findings.*

The original one use chain rule directly, and the edited one calculate sensitivity first.
While they have the same effect, calculate the sensitivity by recursion is clearer mathematically.

### E1Q2
*Use the time package of python and test it on "1_Numpy.py" code and save the 
running time. Change the dtype to torch tensor save the running time as well. 
Comapre the timing results. Explain your findings.*

The running time of the original code is about 0.63s, after changing the dtype to torch.tensor, the running time is 
about 0.52s. The reason may be pytorch has some more efficient design about calculating matrix.

### E1Q3
*Q3: Keep the data size same and change the number of epochs for Q2.
Compare the timing results. Explain your findings.*

|Number of epoch|Time|
|----|----|
|500|0.60|
|1000|1.13|
|1500|1.67|
|2000|2.32|

Time increases when the number of epoch increase, the relationship of increase between these two factors is linear.
Reason: when the running time for each epoch is same, the more epochs we have, the more total running time will be.

### E1Q4
*Q4: Increase the data size and keep the number of epochs for Q2 (Hints: Big number for epochs).
Comapre the timing results. Explain your findings.*

When we double the data size (``` batch_size=128 ```), the running time increase to 4.30s. 
Reason: When the number of epoch is same, increase data size will lead to more computation.

### E1Q5
*Keep the data size big and keep the number of epochs big. Change the dtype to 
torch tensor cuda and compare it with numpy.
Compare the timing results. Explain your findings.*

10000 epoch with cuda: 18.266, without cuda is 10.85.  
I think the running time on CPU is shorter because the data is not complex, and it takes some time to transfer the data
to GPU which slow down the overall speed.

### E2Q1
Q1: Modify  the "1_Numpy.py" file and change the dtype to torch float tensor.
Save the vale of the performance and  plot the followings:

i. Performance index with respect to epochs.
![](./MSE.png)

ii. w1 grad 
![](./GradW1.png)

iii. w2 grad 
![](./GradW2.png)

iv. Check your results. Explain each of your plots.

The Gradient Descent algorithm converges after 10 epochs.   
We can observe that the MSE, gradient of W1 and W2 is close to zero after 10 epochs. Which means the gradient descent
has found the optimum.

## Part 2 NN Module
### E1Q1