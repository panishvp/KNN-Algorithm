# K-Nearest Neighbour-Algorithm

K-Nearest Neighbour is a Lazy learning non-Parametric Algorithm.
Lazy learning means that it does not need any training data for model
generation. All training data is used in the testing phase. This makes
the training phase faster and testing phase slower and costlier. Nonparametric means there no assumption for underlying data
distribution.

How the algorithm works?
Suppose there are two classes ClassA and ClassB which have
elements in it. And there is a new element whose class must be
determined. First, we need to find the closest element near to it. In
the figure we see that the nearest neighbour for the new element is
ClassA so we classify the new element into classA. This is the scenario
when the value of K is 1. Thus, the value of K is varied accordingly and
the accuracy of the algorithm is calculated.

K-NN Implementation:
In Project I have used a random data set which is having many
unknown features and a Target class column which says which
dataset belongs to which class.
I have read the data from the given random data set. I have split
the data using train test split data. Applied KNN at first using k=1.
Created a Confusion matrix and a Classification report.
Later in order to calculate the effective value of k. Algorithm is
repeated by incrementing the K value from 1 to 40 and the error rate
is calculated. A graph is plotted for k value against the error rate in
order to find to which k value does the algorithm can fits in
effectively.