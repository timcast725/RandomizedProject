import sys
import random

def main(argv):
	if len(argv) != 5:
		print("Usage: python3 %s output_file.txt num_packets num_processes field_size" % argv[0])
		return

	output_file = open(argv[1], "w")
	out_string = argv[2] + " " + argv[3] + " " + argv[4] + "\n"
	output_file.write(out_string)
	
	num_packets = int(argv[2])
	num_procs = int(argv[3])
	field_size = int(argv[4])

	x_packets = []
	# Create num_packets new packets
	for i in range(0, num_packets):
		duplicate = True
		x_i = []
		# Continue while randomly generated packet is a duplicate
		while duplicate:
			x_i = [0] * num_packets
			# Generate new packet
			for j in range(0, num_packets):
				x_i[j] = random.randint(1, field_size-1)
		
			# Check if it is a duplicate
			duplicate = False  # in case x_packets is empty
			for x_k in x_packets:
				duplicate = True
				for j in range(0, num_packets):
					if x_i[j] != x_k[j]:
						duplicate = False
						break
				if duplicate:
					break
		x_packets.append(x_i)

	for row in x_packets:
		for i in range(0, num_packets):
			output_file.write(str(row[i]))	
			if i < num_packets-1:
				output_file.write(" ")	
			else:
				output_file.write("\n")	

	output_file.close()

if __name__ == "__main__":
	main(sys.argv)
















