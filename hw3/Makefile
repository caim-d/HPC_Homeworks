CXX = g++
CXXFLAGS = -fopenmp

RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.cpp *.c))

all : $(TARGETS)

%:%.c
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)
