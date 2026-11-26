// Nova's C runtime: what a Nova program compiled through MLIR (--nova-to-llvm) calls for what LLVM IR has no
// instruction for. --convert-nova-to-runtime turns each nova op into a call to one of these.
//
// Each function does exactly what the interpreter does (src/runtime/tensor.cc and interp_tensor.cc): the same
// shapes, the same order of floating-point operations, the same output format and the same runtime errors, so a
// program gives the same output compiled or interpreted. tools/difftest.py checks that, program by program.
//
// A tensor is a NovaTensor*, which MLIR sees as an opaque !llvm.ptr. Its rank and sizes are known only at run time,
// so subscripts and sizes come as a count and up to NOVA_MAX_RANK i64 values (unused ones 0): one entry point per
// op, with no arrays to build in the caller. A function that can fail takes the Nova line number first, for the
// error message. Tensors are not freed yet: the interpreter frees each after its last use, and a pass to do that
// here is still to come.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NOVA_MAX_RANK 6  // as the interpreter's TENSOR_MAX_RANK

typedef struct NovaTensor
{
    int32_t rank;
    int64_t shape[NOVA_MAX_RANK];    // outermost first
    int64_t strides[NOVA_MAX_RANK];  // row-major, in elements; the last is 1
    int64_t rows, cols;              // the 2-D view: cols is the last dimension, rows the product of the others
    int64_t size;                    // the number of elements
    double* data;
} NovaTensor;

// ─── Scalars ────────────────────────────────────────────────────────────────

void nova_print_i32(int32_t x) { printf("%d\n", x); }

void nova_print_f32(float x) { printf("%f\n", x); }  // the float is passed to printf as a double, as in C

void nova_print_f64(double x) { printf("%lf\n", x); }

// ─── Errors ─────────────────────────────────────────────────────────────────

// A runtime error stops the program with the Nova line, as the interpreter's tensor_shape_error does
static void nova_error(int32_t line, const char* message)
{
    fflush(stdout);
    fprintf(stderr, "Runtime error at line %d: %s\n", line, message);
    exit(1);
}

// "(2, 3)", as the interpreter writes a shape in its messages
static void shape_text(const NovaTensor* t, char* out, size_t n)
{
    size_t used = (size_t)snprintf(out, n, "(");
    for (int32_t k = 0; k < t->rank && used < n; k++)
        used += (size_t)snprintf(out + used, n - used, "%s%lld", k ? ", " : "", (long long)t->shape[k]);
    if (used < n) snprintf(out + used, n - used, ")");
}

// Before an integer a / b or a % b (nova.check_divisor): the interpreter's check, where C's division by zero is
// undefined
void nova_check_divisor(int32_t line, int32_t divisor)
{
    if (divisor == 0) nova_error(line, "integer division by zero");
}

// ─── Making tensors ─────────────────────────────────────────────────────────

static NovaTensor* allocate(int32_t rank, const int64_t* shape)
{
    NovaTensor* t = (NovaTensor*)malloc(sizeof(NovaTensor));
    t->rank = rank;
    t->size = 1;
    for (int32_t k = 0; k < rank; k++)
    {
        t->shape[k] = shape[k];
        t->size *= shape[k];
    }
    int64_t stride = 1;
    for (int32_t k = rank - 1; k >= 0; k--)
    {
        t->strides[k] = stride;
        stride *= shape[k];
    }
    t->cols = rank == 0 ? 1 : shape[rank - 1];
    t->rows = 1;
    for (int32_t k = 0; k + 1 < rank; k++) t->rows *= shape[k];
    t->data = (double*)calloc(t->size ? (size_t)t->size : 1, sizeof(double));  // zeros
    return t;
}

static NovaTensor* allocate_like(const NovaTensor* t) { return allocate(t->rank, t->shape); }

static NovaTensor* filled(int32_t line, int32_t rank, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4,
                          int64_t d5, double value)
{
    int64_t shape[NOVA_MAX_RANK] = {d0, d1, d2, d3, d4, d5};
    for (int32_t k = 0; k < rank; k++)
    {
        if (shape[k] < 0)
        {
            char message[64];
            snprintf(message, sizeof message, "negative dimension %lld", (long long)shape[k]);
            nova_error(line, message);
        }
    }
    NovaTensor* t = allocate(rank, shape);
    if (value != 0.0)
        for (int64_t i = 0; i < t->size; i++) t->data[i] = value;
    return t;
}

NovaTensor* nova_zeros(int32_t line, int32_t rank, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4,
                       int64_t d5)
{
    return filled(line, rank, d0, d1, d2, d3, d4, d5, 0.0);
}

NovaTensor* nova_ones(int32_t line, int32_t rank, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4,
                      int64_t d5)
{
    return filled(line, rank, d0, d1, d2, d3, d4, d5, 1.0);
}

// ─── Elements ───────────────────────────────────────────────────────────────

// The flat offset of t[i0, i1, ...], with the interpreter's rank and bounds checks (tensor_offset)
static int64_t offset(int32_t line, const char* op, const NovaTensor* t, int32_t n, const int64_t* idx)
{
    char message[160];
    if (n != t->rank)
    {
        snprintf(message, sizeof message, "%s: a rank-%d tensor needs %d indices, got %d", op, t->rank, t->rank, n);
        nova_error(line, message);
    }
    int64_t off = 0;
    for (int32_t k = 0; k < n; k++)
    {
        if (idx[k] < 0 || idx[k] >= t->shape[k])
        {
            snprintf(message, sizeof message, "%s: index %lld out of range for dimension %d (size %lld)", op,
                     (long long)idx[k], k, (long long)t->shape[k]);
            nova_error(line, message);
        }
        off += idx[k] * t->strides[k];
    }
    return off;
}

double nova_get(int32_t line, NovaTensor* t, int32_t n, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4,
                int64_t i5)
{
    int64_t idx[NOVA_MAX_RANK] = {i0, i1, i2, i3, i4, i5};
    return t->data[offset(line, "tensor_get", t, n, idx)];
}

void nova_set(int32_t line, NovaTensor* t, double value, int32_t n, int64_t i0, int64_t i1, int64_t i2, int64_t i3,
              int64_t i4, int64_t i5)
{
    int64_t idx[NOVA_MAX_RANK] = {i0, i1, i2, i3, i4, i5};
    t->data[offset(line, "tensor_set", t, n, idx)] = value;
}

// ─── Element-wise ───────────────────────────────────────────────────────────

static int same_shape(const NovaTensor* a, const NovaTensor* b)
{
    if (a->rank != b->rank) return 0;
    for (int32_t k = 0; k < a->rank; k++)
        if (a->shape[k] != b->shape[k]) return 0;
    return 1;
}

// Equal shapes; + and - also take an MxN matrix with a 1xN row on the right (require_elementwise)
static void require_elementwise(int32_t line, const char* op, const NovaTensor* a, const NovaTensor* b,
                                int rowBroadcast)
{
    if (same_shape(a, b)) return;
    if (rowBroadcast && a->rank <= 2 && b->rank <= 2 && a->cols == b->cols && b->rows == 1) return;
    char as[96], bs[96], message[256];
    shape_text(a, as, sizeof as);
    shape_text(b, bs, sizeof bs);
    snprintf(message, sizeof message, "'%s' needs tensors of the same shape, got %s and %s", op, as, bs);
    nova_error(line, message);
}

NovaTensor* nova_add(int32_t line, NovaTensor* a, NovaTensor* b)
{
    if (a->rows == 1 && b->rows > 1)  // + commutes: a row on the left moves to the right, the side that broadcasts
    {
        NovaTensor* swap = a;
        a = b;
        b = swap;
    }
    require_elementwise(line, "+", a, b, 1);
    NovaTensor* c = allocate_like(a);
    if (a->rows == b->rows && a->cols == b->cols)
        for (int64_t i = 0; i < a->size; i++) c->data[i] = a->data[i] + b->data[i];
    else  // b is a 1xN row, added to every row of a
        for (int64_t i = 0; i < a->rows; i++)
            for (int64_t j = 0; j < a->cols; j++) c->data[i * a->cols + j] = a->data[i * a->cols + j] + b->data[j];
    return c;
}

NovaTensor* nova_sub(int32_t line, NovaTensor* a, NovaTensor* b)
{
    require_elementwise(line, "-", a, b, 1);
    NovaTensor* c = allocate_like(a);
    if (a->rows == b->rows && a->cols == b->cols)
        for (int64_t i = 0; i < a->size; i++) c->data[i] = a->data[i] - b->data[i];
    else
        for (int64_t i = 0; i < a->rows; i++)
            for (int64_t j = 0; j < a->cols; j++) c->data[i * a->cols + j] = a->data[i * a->cols + j] - b->data[j];
    return c;
}

NovaTensor* nova_mul(int32_t line, NovaTensor* a, NovaTensor* b)
{
    require_elementwise(line, "*", a, b, 0);
    NovaTensor* c = allocate_like(a);
    for (int64_t i = 0; i < a->size; i++) c->data[i] = a->data[i] * b->data[i];
    return c;
}

NovaTensor* nova_div(int32_t line, NovaTensor* a, NovaTensor* b)
{
    require_elementwise(line, "/", a, b, 0);
    NovaTensor* c = allocate_like(a);
    for (int64_t i = 0; i < a->size; i++) c->data[i] = a->data[i] / b->data[i];
    return c;
}

NovaTensor* nova_scale(NovaTensor* t, double k)
{
    NovaTensor* c = allocate_like(t);
    for (int64_t i = 0; i < t->size; i++) c->data[i] = t->data[i] * k;
    return c;
}

NovaTensor* nova_add_scalar(NovaTensor* t, double k)
{
    NovaTensor* c = allocate_like(t);
    for (int64_t i = 0; i < t->size; i++) c->data[i] = t->data[i] + k;
    return c;
}

// ─── Matmul ─────────────────────────────────────────────────────────────────

// The interpreter's matmul_plan: a right side of rank 1 or 2 is one (k x m) matrix every row of the left side
// meets, (..., n, k) @ (k, m); a right side of rank 3 or more is a batched matmul, (b..., n, k) @ (b..., k, m), the
// batch dimensions broadcast as NumPy's do. The products are summed in the same order (i, then t, then j).
NovaTensor* nova_matmul(int32_t line, NovaTensor* a, NovaTensor* b)
{
    char message[160];
    int64_t n, k, m, batches = 1;
    int64_t shape[NOVA_MAX_RANK];
    int32_t rank;
    int64_t aDims[NOVA_MAX_RANK], bDims[NOVA_MAX_RANK], out[NOVA_MAX_RANK];
    int32_t batch = 0;
    if (b->rank <= 2)
    {
        if (a->cols != b->rows)
        {
            snprintf(message, sizeof message, "matmul: inner dimensions differ (%lld vs %lld)", (long long)a->cols,
                     (long long)b->rows);
            nova_error(line, message);
        }
        n = a->rows;
        k = a->cols;
        m = b->cols;
        if (a->rank > 2)
        {
            rank = a->rank;
            memcpy(shape, a->shape, sizeof shape);
        }
        else
        {
            rank = 2;
            shape[0] = a->rows;
        }
        shape[rank - 1] = b->cols;
    }
    else
    {
        if (a->rank < 2) nova_error(line, "matmul: a batched matmul needs a left side of rank 2 or more (got rank 1)");
        int32_t aBatch = a->rank - 2, bBatch = b->rank - 2;
        batch = aBatch > bBatch ? aBatch : bBatch;
        for (int32_t d = 0; d < batch; d++) aDims[d] = bDims[d] = 1;  // lined up from the right
        for (int32_t d = 0; d < aBatch; d++) aDims[batch - aBatch + d] = a->shape[d];
        for (int32_t d = 0; d < bBatch; d++) bDims[batch - bBatch + d] = b->shape[d];
        for (int32_t d = 0; d < batch; d++)
        {
            if (aDims[d] != bDims[d] && aDims[d] != 1 && bDims[d] != 1)
            {
                snprintf(message, sizeof message, "matmul: batch dimension %d differs (%lld vs %lld)", d,
                         (long long)aDims[d], (long long)bDims[d]);
                nova_error(line, message);
            }
            out[d] = aDims[d] > bDims[d] ? aDims[d] : bDims[d];
            batches *= out[d];
        }
        int64_t inner = b->shape[b->rank - 2];
        if (a->cols != inner)
        {
            snprintf(message, sizeof message, "matmul: inner dimensions differ (%lld vs %lld)", (long long)a->cols,
                     (long long)inner);
            nova_error(line, message);
        }
        n = a->shape[a->rank - 2];
        k = a->cols;
        m = b->cols;
        rank = batch + 2;
        for (int32_t d = 0; d < batch; d++) shape[d] = out[d];
        shape[batch] = n;
        shape[batch + 1] = m;
    }
    NovaTensor* c = allocate(rank, shape);
    for (int64_t p = 0; p < batches; p++)
    {
        int64_t ai = 0, bi = 0;
        if (batch > 0)  // which of a's and b's matrices product p uses; a stretched dimension stays at 0
        {
            int64_t rest = p, aStride = 1, bStride = 1;
            for (int32_t d = batch - 1; d >= 0; d--)
            {
                int64_t idx = rest % out[d];
                rest /= out[d];
                if (aDims[d] > 1) ai += idx * aStride;
                if (bDims[d] > 1) bi += idx * bStride;
                aStride *= aDims[d];
                bStride *= bDims[d];
            }
        }
        const double* A = a->data + ai * n * k;
        const double* B = b->data + bi * k * m;
        double* C = c->data + p * n * m;
        for (int64_t i = 0; i < n; i++)
            for (int64_t t = 0; t < k; t++)
                for (int64_t j = 0; j < m; j++) C[i * m + j] += A[i * k + t] * B[t * m + j];
    }
    return c;
}

// ─── Printing ───────────────────────────────────────────────────────────────

// As the interpreter's TEN_PRINT: rank 1 and 2 as [[a, b],\n [c, d]], each element %.4f; a higher-rank tensor as a
// list of its 2-D blocks (the last two dimensions), separated by a blank line
void nova_print_tensor(NovaTensor* t)
{
    int64_t bRows = t->rank >= 2 ? t->shape[t->rank - 2] : 1;
    int64_t bCols = t->cols;
    int64_t nBlocks = (bRows * bCols == 0) ? 0 : t->size / (bRows * bCols);
    if (t->rank > 2) printf("[");
    for (int64_t blk = 0; blk < nBlocks; blk++)
    {
        int64_t base = blk * bRows * bCols;
        if (blk > 0) printf(",\n\n ");
        printf("[");
        for (int64_t i = 0; i < bRows; i++)
        {
            if (bRows > 1) printf("%s", i == 0 ? "[" : (t->rank > 2 ? "  [" : " ["));
            for (int64_t j = 0; j < bCols; j++)
            {
                printf("%.4f", t->data[base + i * bCols + j]);
                if (j < bCols - 1) printf(", ");
            }
            if (bRows > 1) printf("%s", i < bRows - 1 ? "],\n" : "]");
        }
        printf("]");
    }
    if (t->rank > 2) printf("]");
    printf("\n");
}
