# BRUTEFORCE with CUDA
The goal of this project was to practice GPU programming with CUDA. We decided to implement few hacking tool, using bruteforce. The speed improvement made by using the GPU were quite impressive

## RSA cracker
First, a simple RSA breaker
## Prime number generator
In order to test our RSA breaker we decided to implement a prime number generator, as with the Eratosthene sleave is very easy to paralellize
## Hash collision tester
Next we are currently implementing an algorithm to test if an hash algorithm is collision-proof or not. Giving that we cannot use any hash library with CUDA, we implement ourselves a simple hash algorithm.
