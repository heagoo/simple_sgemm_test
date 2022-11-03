CFLAGS = -O2
LDFLAGS = -lmkl_rt
CC = g++

all : sgemm_test igemm_test packed_sgemm_test

sgemm_test : sgemm_test.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o sgemm_test sgemm_test.o

igemm_test : igemm_test.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o igemm_test igemm_test.o

packed_sgemm_test : packed_sgemm_test.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o packed_sgemm_test packed_sgemm_test.o

sgemm_test.o : sgemm_test.cpp
	$(CC) $(CFLAGS) -c sgemm_test.cpp

igemm_test.o : igemm_test.cpp
	$(CC) $(CFLAGS) -c igemm_test.cpp

packed_sgemm_test.o : packed_sgemm_test.cpp
	$(CC) $(CFLAGS) -c packed_sgemm_test.cpp

clean :
	rm -f sgemm_test packed_sgemm_test sgemm_test.o packed_sgemm_test.o
