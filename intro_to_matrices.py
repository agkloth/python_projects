# Introduction to Numerical Python

import numpy as np

# Use the np.array() function to define x and y, which are one-dimensional arrays, i.e. vectors.

x = np.array([3, 4, 5])
y = np.array([4, 9, 7])

print(x+y)

# Two dimensional array

x = np.array([[1, 2], [3, 4]])
print(x)

print(x.ndim)
print(x.dtype)
print(x.shape)
print(x.sum())

# Reshape method with tuple

x = np.array([1, 2, 3, 4, 5, 6])
print('beginning x:\n', x)
x_reshape = x.reshape((2, 3))
print('reshaped x:\n', x_reshape)

# tuple specifies we want two-dimensional array with 2 rows and 3 columns
# \n creates a new line
# method returns a new array with the same elements as x but different shape

# Indexing

# Remember, Python uses 0-based indexing
x_reshape[0, 0] #top-left element of array



# Modifying top-left element of x_reshape

print("x before we modify x_reshape: \n", x)
print("x_shape before we modify x_reshape: \n", x_reshape)
x_reshape[0, 0]=5
print("x_reshape after we modify its top left element: \n", x_reshape)
print("x after we modify top left element of x_reshape: \n", x)

# Remember, elements of tuples cannot be changes, while elements of lists can

# Functions

print(np.sqrt(x))
print(x**2)

# Generating random data

np.random.normal?
# loc = mean
# scale = standard deviation
# size = number of samples

# Generate 50 independent random variables from a N(0,1) distribution.
np.random.seed(42)
x = np.random.normal(size=50)
print(x)

# Create an array y by adding independent N(50,1) random variable to each element of x
y = x + np.random.normal(loc=50, scale=1, size=50)
print(y)


# More functions

print(np.mean(x))
print(np.var(x))
print(np.std(x))

# Using rng.normal instead of np.random

rng = np.random.default_rng(1303)
print(rng.normal(scale=5, size=2))
rng2 = np.random.default_rng(1303)
print(rng2.normal(scale=5, size=2))

# Get the same result

X = rng.standard_normal((10,3))
print(X)

# Graphics and Plots

from matplotlib.pyplot import subplots, show
fig, ax = subplots(figsize=(8, 8))
x = rng.standard_normal(100)
y = rng.standard_normal(100)
ax.plot(x, y);
show()

# fig: overall figure container
# ax: subplot where you will draw your data.
# Note: we have unpacked the tuple of length two returned by subplots() into the two distinct variables fig and ax

# Create scatterplot

fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker='o');
show()

# semi-colon prevents text

# Labeling plots

fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker='o')
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y");
show()

# Creating several plots within a figure
fig, axes = subplots(nrows=2, ncols=3,
figsize=(15, 5));
show()

# produce scatter plot with 'o' in the second column of the first row
# produce scatter plot with '+' in the third column of the second row
axes[0,1].plot(x, y, 'o')
axes[1,2].scatter(x, y, marker='+')
fig
show()


# Sequences and Slice Notation
import numpy as np
sq1 = np.linspace(0, 10, 11)
print(sq1)
#linspace can be used to create. a sequence of numbers

#np.arange() function; returns a sequence of numbers spaced out by step
seq2 = np.arange(0,10)
print(seq2)
# when slicing and sequences, the indexing starts at 1

# Indexing Data

import numpy as np
A = np.array(np.arange(16)).reshape(4,4)
print(A)

print(A[1,2])

print(A[[1,3]])
# using a list to print the second and fourth row

print(A[:,[0,2]])
# this will print the first and third column while specifying ":" to give all the rows

print(A[1:4:2,0:3:2])
# slice 1:4:2 captures second and fourth items of a sequence
# slice 0:3:2 captures the first and third items
# we cannot retrieve a submatrix using lists

