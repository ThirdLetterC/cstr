# Memory and API contracts

[Back to README](../README.md) · [API guide](api.md)

## String views

A `cs_view` is a read-only view of bytes: `{ const char *data; size_t len; }`. It does not own or copy the bytes. Its length is in bytes, and it does not need a trailing NUL byte.

`{NULL, 0}` is a valid empty view. A view with `data == NULL` and `len > 0` is invalid and returns `CS_EINVAL`. Operations do not pass null pointers to `memcpy` or `memcmp`, even for zero-length input.

## Owned strings

A `cs_string` owns a buffer and stores `data`, `len`, `cap`, `alloc`, `generation`, and `flags`. Its layout does not change with `CS_ENABLE_*` or `CS_NO_HEAP`. Its size depends on the platform; use `sizeof(cs_string)` when you need the size.

`cap` counts the bytes available for content. It excludes the trailing NUL byte, so an allocated buffer has `cap + 1` bytes. An initialized string always has `data[len] == '\0'`.

A newly initialized or reset empty string has `cap == 0`. Its `data` points to an internal, read-only empty string. This means `cs_c_str()` always returns a non-null pointer for an initialized string.

Do not copy a `cs_string` with assignment or `memcpy`. Use `cs_clone(&dst, &src)` to copy its content, or `cs_move(&dst, &src)` to transfer ownership. After a move, `src` is a valid empty string with `cap == 0`.

Calling `cs_destroy(&s)` or `cs_reset(&s)` more than once is safe. `cs_detach()` gives the allocated buffer to the caller. `cs_adopt()` takes ownership of a buffer and requires an explicitly supplied, compatible `cs_allocator`.

## Fixed buffers

A `cs_buffer` uses storage supplied by the caller: `{ char *data; size_t len; size_t cap; }`. It never allocates heap memory.

`cs_buffer_init(b, storage, storage_size)` sets `cap` to `storage_size - 1` and maintains `data[len] == '\0'`.

If an operation needs more space, it returns `CS_EBUFFER`, keeps the destination unchanged, and writes the required buffer size to `*out_required`. That size includes the trailing NUL byte.

## Allocators

You can supply a custom `cs_allocator`. The requested alignment, `align`, is a power of two. The allocator must outlive the objects that use it.

The library also supports scratch storage through `cs_workspace`, and a default allocator that you can change with `cs_set_default_allocator()` and restore with `cs_reset_default_allocator()`.

For optional mimalloc support, build with `CS_USE_MIMALLOC=1` and use `cs_mimalloc_allocator(mi_heap_t *)`.

## Views, iterators, and indexes after changes

Changing or destroying a string increments its `generation` counter. Treat views, iterators, and indexes made from that string as invalid after a change.

`cs_index_is_valid()` checks whether an index still matches its owner's generation. A plain `cs_view` only stores a pointer and length, so it cannot detect invalidation itself. The caller must keep its source alive and avoid using stale views or iterators.

## Input that overlaps the destination

Operations such as `cs_append`, `cs_prepend`, `cs_insert_bytes`, `cs_replace_range_bytes`, `cs_assign`, `cs_repeat`, and `cs_append_many` support input taken from the destination's own storage.

Before reallocating, these operations check whether the input pointer is inside `[s->data, s->data + s->cap]`. They preserve the input by tracking its offset or using temporary storage.

## Errors

On an error, the destination `cs_string` or `cs_buffer` keeps its previous logical content, length, trailing NUL byte, and memory ownership. Errors include:

| Status | Meaning |
| :--- | :--- |
| `CS_EINVAL` | Invalid argument. |
| `CS_ENOMEM` | Memory allocation failed. |
| `CS_EBUFFER` | The supplied buffer is too small. |
| `CS_EUTF8` | Invalid UTF-8 in strict mode. |
| `CS_ERANGE` | An index or value is outside the allowed range. |
| `CS_EOVERFLOW` | Size or capacity arithmetic overflowed. |
| `CS_EUNSUPPORTED` | A feature is disabled or unsupported. |
| `CS_ELIMIT` | A configured resource limit was reached. |
| `CS_ESYNTAX` | Invalid syntax in a format, pattern, or escape sequence. |

## Thread safety

Several threads may read the same `cs_view` or `const cs_string` at once. Threads may also change separate objects at the same time. Coordinate access when one thread changes an object that another thread uses.

CPU backend selection uses atomic initialization with `atomic_compare_exchange_strong` to avoid data races.

## License

Library code and algorithms use MIT. Bundled Unicode 17.0.0 tables use `Unicode-3.0` (Unicode License v3).
