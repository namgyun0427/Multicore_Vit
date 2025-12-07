CC = gcc
CFLAGS = -std=gnu17 -pedantic-errors -Wall -Wextra -Werror -Ilinux
LIBS = -lm -lOpenCL

################################################################################################
IN_DIR_TEST = tests

OUT_DIR_BUILD = build
OUT_DIR_OBJ = $(OUT_DIR_BUILD)/obj
OUT_DIR_TEST = $(OUT_DIR_BUILD)/tests

################################################################################################
IN_OBJ = \
	linux/posix_port.c \
	comparator.c \
	Network.c \
	ViT_seq.c \
	ViT_cl.c \
	v_multihead_attn.c \
	v_layer_norm.c \
	v_linear_layer.c \
	v_mlp_block.c

OUT_OBJ = $(patsubst %.c,$(OUT_DIR_OBJ)/%.o,$(IN_OBJ))


IN_BIN = main.c

OUT_BIN = $(OUT_DIR_BUILD)/main


IN_TEST = \
	$(IN_DIR_TEST)/v_reduce_sum.c
	
OUT_TEST = $(patsubst $(IN_DIR_TEST)/%.c,$(OUT_DIR_TEST)/%,$(IN_TEST))


################################################################################################
all: obj bin test



obj: $(OUT_OBJ)

$(OUT_DIR_OBJ)/%.o: %.c | $(OUT_DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUT_DIR_OBJ): 
	mkdir -p $@
	mkdir -p $@/linux



bin: obj $(OUT_BIN)

$(OUT_BIN): $(IN_BIN) $(OUT_OBJ) | $(OUT_DIR_BUILD)
	$(CC) $(CFLAGS) $(OUT_OBJ) $< -o $@ $(LIBS)



test: obj $(OUT_TEST)

$(OUT_DIR_TEST)/%: $(IN_DIR_TEST)/%.c $(OUT_OBJ) | $(OUT_DIR_TEST)
	$(CC) $(CFLAGS) $(OUT_OBJ) $< -o $@ $(LIBS)


$(OUT_DIR_TEST):
	mkdir -p $@



$(OUT_DIR_BUILD):
	mkdir -p $@



clean:
	rm -rf $(OUT_DIR_BUILD)



.PHONY: all obj bin test clean
