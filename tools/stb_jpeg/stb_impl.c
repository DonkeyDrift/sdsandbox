#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *data; size_t len, cap; } stbj_buf;

static void stbj_cb(void *ctx, void *data, int size) {
    stbj_buf *b = (stbj_buf *)ctx;
    if (b->len + (size_t)size > b->cap) {
        size_t ncap = b->cap ? b->cap : 65536;
        while (ncap < b->len + (size_t)size) ncap *= 2;
        b->data = (unsigned char *)realloc(b->data, ncap);
        b->cap = ncap;
    }
    memcpy(b->data + b->len, data, (size_t)size);
    b->len += (size_t)size;
}

/* rgb: 3 bytes/px。成功返回长度并置 *out(malloc,用 stbj_free 释放),失败返回 0 */
int stbj_write_jpg(const unsigned char *rgb, int w, int h, int quality, unsigned char **out) {
    stbj_buf b; b.data = 0; b.len = 0; b.cap = 0;
    if (!stbi_write_jpg_to_func(stbj_cb, &b, w, h, 3, (const void *)rgb, quality)) {
        free(b.data); return 0;
    }
    *out = b.data;
    return (int)b.len;
}

void stbj_free(void *p) { free(p); }
