Randomized Algorithms Project

Paper: A Randomized Algorithm and Performance
Bounds for Coded Cooperative Data Exchange
Alex Sprintson, Parastoo Sadeghi, Graham Booker, and Salim El Rouayheb

Files:

main.py:
	Main python file for running experiments pertaining to the
RDE algorithm. It will count the number of bytes of the Naive Algorithm
vs the RDE algorithm. The Naive algorithm consists of having each process
send all its known packets to all other processes. The RDE algorithm uses
linear coding to try and reduce the number of bytes sent.

To run:
	python3 main.py input.txt [run_type]

The input.txt should be formatted based on input_example.txt. The run_type
can be either "missing_one" or "all_but_one". If left blank, missing_one is
the default. 

input_example.txt
	File example for how to format an input file for main.py

require.sh
	Installs dependencies for Python to run main. Requires that Python 
3.5 or later and pip verion >= 9.0.1 are already installed.


