CC = gcc
CFLAGS = -std=gnu17 --pedantic-errors -Wall -Wextra
LIBS = -lm -lOpenCL

DIR_BUILD = build
DIR_OBJ = $(DIR_BUILD)/obj
DIR_TEST = $(DIR_BUILD)/tests

IN_SRC = posix_port.c comparator.c Network.c ViT_seq.c ViT_cl.c Main.c
OUT_OBJ = $(patsubst %.c, $(DIR_OBJ)/%.o, $(IN_SRC))

MAIN_OBJ = Main.o
MAIN_BIN = $(DIR_BUILD)/main


all: $(MAIN_BIN)
	@echo $(OUT_OBJ)

$(MAIN_BIN): $(OUT_OBJ)
	$(CC) $(CFLAGS) $(OUT_OBJ) -o $@ $(LIBS)

$(DIR_OBJ)/%.o: %.c | $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(DIR_OBJ): 
	mkdir -p $@

test:
	$(CC) $(CFLAGS)  build/obj/posix_port.o  build/obj/comparator.o  build/obj/Network.o  build/obj/ViT_seq.o  build/obj/ViT_cl.o tests/reduce_sum.c -o test $(LIBS)

clean:
	rm -rf $(DIR_BUILD)

.PHONEY = all clean