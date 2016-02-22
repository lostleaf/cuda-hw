#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint8_t *d_bins, uint32_t *d_input, int *d_bins_32, int input_width, int input_height, int bins_size);

/* Include below the function headers of any other functions that you implement */

void init(uint8_t **d_bins, uint32_t **d_input, int **d_bins_32, int hist_width, int hist_height, 
    int input_width, int input_height, uint32_t *h_input);

void final(uint8_t *d_bins, uint32_t *d_input, int *d_bins_32, uint8_t *h_bins, int bins_size);
#endif
