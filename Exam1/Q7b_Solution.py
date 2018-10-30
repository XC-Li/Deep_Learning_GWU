"""Machine Learning 2 Section 10 @ GWU
Exam 1 - Solution for Q7 Driver
Author: Xiaochi (George) Li"""

try:
    from Q7a_Solution import helper
except:
    print("Error! Needs to put in the same directory with Q7a_Solution.py")
    exit(-1)
import os
import numpy as np
trail_length = 10
trail_list = []
for i in range(trail_length):
    file_name = "trace" + str(i) + ".csv"
    if os.path.isfile(file_name):
        os.remove(file_name)  # if exist the file, then remove it
    trail_list.append(helper(file_name))  # call the Neural Network and get average gradient and save csv file

trail = np.concatenate(trail_list, axis=0)
print("The shape of trail matrix:", trail.shape)
trail_avg = np.average(trail, axis=0)
trail_std = np.std(trail, axis=0)

max_ten_avg = trail_avg.argsort(kind='quicksort')[-10:][::-1]
max_ten_std = trail_std.argsort(kind='quicksort')[-10:][::-1]

print("The top ten input(Pictures) that has largest average of gradient:\n", max_ten_avg)
print("The top ten input(Pictures) that has largest standard deviation of gradient:\n", max_ten_std)