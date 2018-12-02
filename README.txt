Randomized Algorithms Project

Paper: A Randomized Algorithm and Performance
Bounds for Coded Cooperative Data Exchange
Alex Sprintson, Parastoo Sadeghi, Graham Booker, and Salim El Rouayheb

Files:

main_exps.py:
	Main python file for running experiments pertaining to the
RDE algorithm. It will count the number of bytes of the Naive Algorithm
vs the RDE algorithm. The Naive algorithm consists of having each process
send all its known packets to all other processes. The RDE algorithm uses
linear coding to try and reduce the number of bytes sent.
    Requires to be run with SageMath, see require.txt

To run:
	sage -python main.py input.txt [run_type]

The input.txt should be formatted based on input_example.txt. The run_type
can be missing_one, has_one, some, half, most, random_10, random_25,
random_50, random_75, or random_90. If left blank, missing_one is
the default.

input_example.txt
	File example for how to format an input file for main.py

require.txt
    Has information about installing SageMath and using it.

generate_input.py
    Creates an input for main.py. Randomly chooses values for X using field_size.

To run:
    python3 generate_input.py <output.txt> <num_packets> <num_clients> <field_size>
