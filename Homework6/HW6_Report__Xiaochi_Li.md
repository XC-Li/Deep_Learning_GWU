## Report for homework 6
Author: Xiaochi Li

## Part 1 pytorch Basics
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