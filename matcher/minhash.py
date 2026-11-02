"""MinHash bands over title tokens, bit-for-bit as Spark SQL computes them (`xxhash64`, seed 42), so a record and a work
share a band here exactly when they would in production: 12 MinHash values, 6 bands of 2."""
from . import text

M64 = (1 << 64) - 1
P1, P2, P3, P4, P5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5)
SEED = 42


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def _fmix(h):
    h ^= h >> 33; h = (h * P2) & M64
    h ^= h >> 29; h = (h * P3) & M64
    return h ^ (h >> 32)


def _round(acc, v):
    return (_rotl((acc + v * P2) & M64, 31) * P1) & M64


def _merge(h, v):
    return ((h ^ _round(0, v)) * P1 + P4) & M64


def hash_bytes(b, seed):
    n, i = len(b), 0
    if n >= 32:
        v1, v2, v3, v4 = (seed + P1 + P2) & M64, (seed + P2) & M64, seed, (seed - P1) & M64
        while i <= n - 32:
            v1 = _round(v1, int.from_bytes(b[i:i + 8], "little")); v2 = _round(v2, int.from_bytes(b[i + 8:i + 16], "little"))
            v3 = _round(v3, int.from_bytes(b[i + 16:i + 24], "little")); v4 = _round(v4, int.from_bytes(b[i + 24:i + 32], "little"))
            i += 32
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & M64
        for v in (v1, v2, v3, v4):
            h = _merge(h, v)
    else:
        h = (seed + P5) & M64
    h = (h + n) & M64
    while i <= n - 8:
        h ^= _round(0, int.from_bytes(b[i:i + 8], "little"))
        h = (_rotl(h, 27) * P1 + P4) & M64
        i += 8
    if i <= n - 4:
        h ^= (int.from_bytes(b[i:i + 4], "little") * P1) & M64
        h = (_rotl(h, 23) * P2 + P3) & M64
        i += 4
    while i < n:
        h ^= (b[i] * P5) & M64
        h = (_rotl(h, 11) * P1) & M64
        i += 1
    return _fmix(h)


def hash_int(v, seed):
    h = (seed + P5 + 4) & M64
    h ^= ((v & 0xFFFFFFFF) * P1) & M64
    return _fmix((_rotl(h, 23) * P2 + P3) & M64)


def hash_long(v, seed):
    h = (seed + P5 + 8) & M64
    h ^= _round(0, v & M64)
    return _fmix((_rotl(h, 27) * P1 + P4) & M64)


def _signed(h):
    return h - (1 << 64) if h >= 1 << 63 else h


def xxhash64_str_int(s, k):
    """xxhash64(<string>, <int>) in Spark SQL."""
    return _signed(hash_int(k, hash_bytes(s.encode("utf-8"), SEED)))


def xxhash64_int_long_long(i, a, b):
    """xxhash64(<int>, <bigint>, <bigint>) in Spark SQL."""
    return _signed(hash_long(b, hash_long(a, hash_int(i, SEED))))


def bands(nt):
    """The title's 6 bands, or None when it has fewer than 3 usable tokens."""
    mt = text.title_tokens(nt)
    if len(mt) < 3:
        return None
    mh = [min(xxhash64_str_int(t, k) for t in mt) for k in range(12)]
    return [xxhash64_int_long_long(i, mh[2 * i], mh[2 * i + 1]) for i in range(6)]
