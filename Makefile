CC = gcc
CFLAGS = -std=gnu17 --pedantic-errors -Wall -Wextra
LIBS = -lm -lOpenCL

################################################################################################
IN_DIR_TEST = tests

OUT_DIR_BUILD = build
OUT_DIR_OBJ = $(OUT_DIR_BUILD)/obj
OUT_DIR_TEST = $(OUT_DIR_BUILD)/tests

################################################################################################
IN_SRC = posix_port.c comparator.c Network.c ViT_seq.c ViT_cl.c Main.c
OUT_OBJ = $(patsubst %.c, $(OUT_DIR_OBJ)/%.o, $(IN_SRC))

IN_BIN = Main.c
OUT_BIN = $(OUT_DIR_BUILD)/main

IN_TEST = 

MAIN_OBJ = Main.o
MAIN_BIN = $(OUT_DIR_BUILD)/main

IN_TEST = 


################################################################################################

all: $(MAIN_BIN)
	@echo $(OUT_OBJ)

$(MAIN_BIN): $(OUT_OBJ)
	$(CC) $(CFLAGS) $(OUT_OBJ) -o $@ $(LIBS)

$(OUT_DIR_OBJ)/%.o: %.c | $(OUT_DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUT_DIR_OBJ): 
	mkdir -p $@

test:
	$(CC) $(CFLAGS)  build/obj/posix_port.o  build/obj/comparator.o  build/obj/Network.o  build/obj/ViT_seq.o  build/obj/ViT_cl.o tests/reduce_sum.c -o test $(LIBS)

clean:
	rm -rf $(OUT_DIR_BUILD)

.PHONEY = all clean