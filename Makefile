CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
CLANG_FORMAT ?= clang-format
FORMAT_SOURCES := cstr.c cstr.h test_cstr.c bench_cstr.c

.PHONY: all test test-asan bench format clean

all: test_cstr bench_cstr

test_cstr: cstr.c cstr.h test_cstr.c
	$(CC) $(CFLAGS) cstr.c test_cstr.c -lm -o test_cstr

bench_cstr: cstr.c cstr.h bench_cstr.c
	$(CC) $(CFLAGS) cstr.c bench_cstr.c -lm -o bench_cstr

test: test_cstr
	./test_cstr

test-asan: cstr.c cstr.h test_cstr.c
	$(CC) -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Werror -pedantic cstr.c test_cstr.c -lm -o test_cstr_asan
	./test_cstr_asan

bench: bench_cstr
	./bench_cstr

format:
	$(CLANG_FORMAT) -i $(FORMAT_SOURCES)

clean:
	rm -f test_cstr test_cstr_asan bench_cstr *.o
