CFLAGS = -O2
LDFLAGS = -lmkl_rt
CC = g++

all : sgemm_test packed_sgemm_test

sgemm_test : sgemm_test.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o sgemm_test sgemm_test.o

packed_sgemm_test : packed_sgemm_test.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o packed_sgemm_test packed_sgemm_test.o

sgemm_test.o : sgemm_test.cpp
	$(CC) $(CFLAGS) -c sgemm_test.cpp

packed_sgemm_test.o : packed_sgemm_test.cpp
	$(CC) $(CFLAGS) -c packed_sgemm_test.cpp

clean :
	rm -f sgemm_test packed_sgemm_test sgemm_test.o packed_sgemm_test.o
