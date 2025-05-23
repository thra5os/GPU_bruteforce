# BRUTEFORCE with CUDA
The aim of this project was to practice GPU programming with CUDA. We decided to implement some hacking tools, using bruteforce. The improvement of execution time made using the GPU were quite impressive.

## RSA cracker
First, we did a simple RSA breaker, but it worked only with small numbers (up to 64 bit integers) so we decided to use a different method to manipulate thoses numbers, first with the CUDA library CUMP and XMP, but we found a simplier alternative by simply storing those numbers into arrays.
We improved the research of the factors of n with the Prime Number Generator.

## Prime number generator
In order to test our RSA breaker we decided to implement a prime number generator, as with the Eratosthene sleave is very easy to paralellize.

## Hash collision tester
Next we implemented an algorithm to test if an hash algorithm is collision-proof or not. Giving that we cannot use any hash library with CUDA, we implemented ourselves a simple hash algorithm, then we used the DJB2 hash algorothm to improve the hash (-> less collisions)

