# Change the path based on your installation folder.

# -- default
# CPX_PATH = ~/local/opt/ibm/ILOG/CPLEX_Studio2211/cplex/
# CC = g++
# ARCH = x86-64_linux

# -- arm64_osx
CPX_PATH = /Applications/CPLEX_Studio2211/cplex
CC = g++
ARCH = arm64_osx

cpx:
	$(CC) -g -L $(CPX_PATH)/lib/$(ARCH)/static_pic/ -I $(CPX_PATH)/include/ilcplex/ -o get_cuts get_cuts.cpp -lm -lcplex -ldl -lpthread

