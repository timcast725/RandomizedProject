import sys
import numpy as np
import sympy
from scipy.linalg import solve
import math
import random

def mod_matrix(matrix, field_size):
    """ Mods each element of matrix with field_size """
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            matrix[i][j] = math.floor(matrix[i][j]) % field_size

def reduce_matrix(matrix):
    """ Remove columns to return only linearly independent columns """
    _, inds = sympy.Matrix(matrix.T).T.rref()
    cols_to_delete = []
    for i in range(0, len(matrix[0])):
        if i not in inds:
            cols_to_delete.append(i)

    for i in cols_to_delete:
        matrix = np.delete(matrix, i, 1)
    return matrix

class process():
    """ Defines a single mobile client in our algorithm

    Each mobile client in the algorithm keeps track of 
    their subset of x and the p's and gamma's received at 
    each time step.
    """
    def __init__(self, pid, x_subset, u_vector):
        """ Default constructor """
        self.pid = pid
        self.x_subset = x_subset.copy()
        self.p_set = np.array([])
        self.gamma_set = np.array([])
        self.u_vector = u_vector.copy()

    def copy(self, other):
        self.pid = other.pid
        self.x_subset = other.x_subset.copy()
        self.p_set = other.p_set.copy()
        self.gamma_set = other.gamma_set.copy()
        self.u_vector = other.u_vector.copy()
        

    def to_string(self):
        """ Returns a string with all member variables """
        string = "PID = " + str(self.pid) + "\n"
        string += "x_subset = " + str(self.x_subset) + "\n"
        string += "u_vector = " + str(self.u_vector) + "\n"
        string += "p_set = " + str(self.p_set) + "\n"
        string += "gamma_set = " + str(self.gamma_set) + "\n\n"
        return string

    def send_all(self, processes):
        """ Used by the naive algorithm 

        Function simulates sending to all processes.
        In actuality, it just counts how many bits would have to
        be sent for this process to broadcast all its packets
        """
        bits_used = 0
        for codeword in self.x_subset:
            for num in codeword:
                bits_used += len("{0:b}".format(num))
        return bits_used    

    def rank(self):
        """ Returns rank of u_vector U gamma_set """
        # If nothing in gamma_set yet, use u_vector
        if len(self.gamma_set) == 0:
            rank = 0
            for val in self.u_vector:
                if val != 0:
                    rank += 1
            return rank

        # Otherwise, append u_vector to gamma and get rank
        u_gamma = np.append(self.gamma_set, np.array([np.array(self.u_vector)]).T, axis=1)
        return np.linalg.matrix_rank(u_gamma)

    def update_u_vec(self, processes):
        """ Used by less naive algorithm

        Updates each processes u_vector to reflect what was sent
        during this round by self process.
        """
        for proc in processes:
            for i in range(0, len(self.u_vector)):
                if self.u_vector[i] == 1:
                    proc.u_vector[i] = 1

    def generate_gamma(self, field_size):
        """ Generates a new encoding vector to send to others """
        gamma = [0] * len(self.u_vector)
        for i in range(0, len(self.u_vector)):
            if self.u_vector[i] != 0:
                gamma[i] = random.randint(1, field_size-1)
        return gamma

    def rde_broadcast(self, processes, gamma, field_size):
        """ Sends encoding vector to all other processes 

        Also calculates p_i for this round for each process to
        add to their matrix p. Returns the number of bits used
        by this broadcast.
        """
        bits_used = 0

        # Create p_i for this round
        p = np.array([[0] * len(self.x_subset[0])])
        j = 0
        for i in range(0, len(gamma)):
            if gamma[i] != 0:
                bits_used += len("{0:b}".format(gamma[i]))
                p = np.add(p, np.multiply(gamma[i], self.x_subset[j]))
                j += 1

        mod_matrix(p, field_size)

        # Count bits in p 
        for row in p:
            for col in row:
                bits_used += len("{0:b}".format(col))

        # Broadcast p and gamma to all processes
        for proc in processes:
            if(len(proc.p_set) == 0):
                proc.p_set = p.T
                proc.gamma_set = np.array([np.array(gamma)]).T
            else:
                proc.p_set = np.append(proc.p_set, p.T, axis=1)
                proc.gamma_set = np.append(proc.gamma_set, np.array([np.array(gamma)]).T, axis=1)
            
        return bits_used

    

def naive_algorithm(x, processes):
    """ Counts the number of bits used by the Naive algorithm """
    bits_used = 0
    for proc in processes:
        bits_used += proc.send_all(processes)
    return bits_used

def less_naive_algorithm(x, processes):
    """ Counts the number of bits used by the Naive algorithm """
    bits_used = 0
    proc_max = processes[0]

    # Continue until all processes can recalculate the original X
    all_packets_received = False
    i = 0
    while not all_packets_received:
        i += 1
        # Determine the process with max rank for set of received x_i's
        for proc in processes:
            if proc.rank() > proc_max.rank():
                proc_max = proc

        if proc_max.rank() == 0:
            print("ERROR: no one has any packets")
            return 0

        # send to all processes
        bits_used += proc_max.send_all(processes)  
        proc_max.update_u_vec(processes)

        # See if all processes can calculate X
        all_packets_received = True
        for proc in processes:
            if proc.rank() < len(x):
                all_packets_received = False 

    return bits_used


def rde_algorithm(x, processes, field_size):
    """ Runs RDE algorithm and returns bits used """
    bits_used = 0
    proc_max = processes[0]

    # Continue until all processes can recalculate the original X
    all_packets_received = False
    i = 0
    while not all_packets_received:
        i += 1
        print("Round " + str(i))
        # Determine the process with max rank for set of received x_i's
        for proc in processes:
            if proc.rank() > proc_max.rank():
                proc_max = proc

        if proc_max.rank() == 0:
            print("ERROR: no one has any packets")
            return 0

        # generate encoding vector
        gamma_i = proc_max.generate_gamma(field_size)  
        # send to all processes
        bits_used += proc_max.rde_broadcast(processes, gamma_i, field_size)  

        # See if all processes can calculate X
        all_packets_received = True
        for proc in processes:
            if proc.rank() < len(x):
                all_packets_received = False 

    print("%d rounds taken" % i)
    return bits_used    


def main(argv):
    if(len(argv) != 2 and len(argv) != 3):
        print("Usage: python3 %s input.txt [run_type]" % argv[0])
        print("Run types allowed: missing_one, has_one, some, half, most, random_10, random_25, random_50, random_75, random_90")
        return    

    random.seed(a=None, version=2)
    run_type = ""  # Determines how many packets each process gets 
    if len(argv) == 3: 
        run_type = argv[2]
    else:
        run_type = "missing_one"

    # Parse input file
    input_file = open(argv[1], "r")    
    x = []
    i = 0
    field_size = 0  # Finite field size, q in paper
    num_packets = 0  # number of packets in X, n in paper
    num_processes = 0  # number of processes, k in paper
    for line in input_file.readlines():
        if i == 0:
            # The first line of the input file
            # gives us the number of packets,
            # the number of processes
            # and the field size.
            args = line.split(" ")
            num_packets = int(args[0])
            num_processes = int(args[1])
            field_size = int(args[2])
        else:
            x_i = line.split(" ")
            x_i = list(map(int, x_i))
            x.append(x_i)
        i += 1
    input_file.close()

    # Create processes
    processes = []
    for i in range(0, num_processes):
        x_subset = []
        u_vector = [0] * num_packets
        if run_type == "missing_one":
            # Each process has all but one codeword
            for j in range(0, num_packets):
                if i != j:
                    x_subset.append(x[j])
                    u_vector[j] = 1
        if run_type == "has_one":
            # Each process has only 1 codeword
            x_subset.append(x[i])
            u_vector[i] = 1
        if run_type == "some":
            # Each process has quarter of codewords
            for j in range(-1*math.ceil(num_packets/8.0), math.ceil(num_packets/8.0)):
                ind = (i + j) % num_packets
                x_subset.append(x[ind])
                u_vector[ind] = 1
        if run_type == "half":
            # Each process has quarter of codewords
            for j in range(-1*math.ceil(num_packets/4.0), math.ceil(num_packets/4.0)):
                ind = (i + j) % num_packets
                x_subset.append(x[ind])
                u_vector[ind] = 1
        if run_type == "most":
            # Each process has quarter of codewords
            for j in range(-1*math.ceil(3*num_packets/8.0), math.ceil(3*num_packets/8.0)):
                ind = (i + j) % num_packets
                x_subset.append(x[ind])
                u_vector[ind] = 1
        if "random" in run_type:
            # Gets a packet with random chance
            for j in range(0, num_packets):
                take_packet = False
                if run_type == "random_10":
                    if random.randint(1, 10) == 1:
                        take_packet = True
                if run_type == "random_25":
                    if random.randint(1, 4) == 1:
                        take_packet = True
                if run_type == "random_50":
                    if random.randint(0, 1) == 1:
                        take_packet = True
                if run_type == "random_75":
                    if random.randint(1, 4) > 1:
                        take_packet = True
                if run_type == "random_90":
                    if random.randint(1, 10) > 1:
                        take_packet = True
                if take_packet:
                    x_subset.append(x[j])
                    u_vector[j] = 1
            

        # Create process and add to list
        new_process = process(i, x_subset, u_vector)
        processes.append(new_process)

    naive_bits_used = naive_algorithm(x, processes)
    print("Used %d bits using the Naive Algorithm" % naive_bits_used)

    processes_copy = []
    for proc in processes:
        new_proc = process(0, x_subset, u_vector)
        new_proc.copy(proc)
        processes_copy.append(new_proc)
        
    less_naive_bits_used = less_naive_algorithm(x, processes_copy)
    print("Used %d bits using the Less Naive Algorithm" % less_naive_bits_used)

    rde_bits_used = rde_algorithm(x, processes, field_size)
    print("Used %d bits using the RDE Algorithm" % rde_bits_used)


if __name__ == "__main__":
    main(sys.argv)






























