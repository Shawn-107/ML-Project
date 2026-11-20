"""Minimal read-only parquet reader (numpy + pandas only).
Supports SNAPPY/UNCOMPRESSED, PLAIN and dictionary encodings, data page v1, flat columns.
Usage: python -I mini_parquet.py <in.parquet> <out.csv>"""
import sys, struct
import numpy as np, pandas as pd


class R:  # Thrift compact protocol reader
    def __init__(s, b, i=0): s.b, s.i = b, i
    def byte(s):
        v = s.b[s.i]; s.i += 1; return v
    def varint(s):
        r = sh = 0
        while True:
            c = s.byte(); r |= (c & 0x7F) << sh
            if not c & 0x80: return r
            sh += 7
    def zz(s):
        n = s.varint(); return (n >> 1) ^ -(n & 1)
    def binary(s):
        n = s.varint(); v = s.b[s.i:s.i + n]; s.i += n; return v
    def value(s, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3: return s.byte()
        if t in (4, 5, 6): return s.zz()
        if t == 7: v = struct.unpack('<d', s.b[s.i:s.i + 8])[0]; s.i += 8; return v
        if t == 8: return s.binary()
        if t in (9, 10):
            h = s.byte(); n = h >> 4; et = h & 15
            if n == 15: n = s.varint()
            return [s.elem(et) for _ in range(n)]
        if t == 11:
            n = s.varint()
            if n == 0: return {}
            kv = s.byte(); kt, vt = kv >> 4, kv & 15
            return {s.elem(kt): s.elem(vt) for _ in range(n)}
        if t == 12: return s.struct()
        raise ValueError(f"bad thrift type {t}")
    def elem(s, t):
        if t in (1, 2): return bool(s.byte() == 1)
        return s.value(t)
    def struct(s):
        d, fid = {}, 0
        while True:
            h = s.byte()
            if h == 0: return d
            t, delta = h & 15, h >> 4
            fid = fid + delta if delta else s.zz()
            d[fid] = s.value(t)


def snappy(b):
    r = R(b); n = r.varint(); out = bytearray(); i = r.i
    while i < len(b):
        tag = b[i]; i += 1; k = tag & 3
        if k == 0:
            ln = tag >> 2
            if ln < 60: ln += 1
            else:
                nb = ln - 59; ln = int.from_bytes(b[i:i + nb], 'little') + 1; i += nb
            out += b[i:i + ln]; i += ln
        else:
            if k == 1: ln = 4 + ((tag >> 2) & 7); off = ((tag >> 5) << 8) | b[i]; i += 1
            elif k == 2: ln = 1 + (tag >> 2); off = int.from_bytes(b[i:i + 2], 'little'); i += 2
            else: ln = 1 + (tag >> 2); off = int.from_bytes(b[i:i + 4], 'little'); i += 4
            for _ in range(ln): out.append(out[-off])
    assert len(out) == n, (len(out), n)
    return bytes(out)


def rle_hybrid(buf, bw, count):
    """Decode RLE/bit-packed hybrid into a numpy array of `count` ints."""
    r = R(buf); out = []; got = 0
    while got < count:
        h = r.varint()
        if h & 1:
            n = (h >> 1) * 8; nbytes = (h >> 1) * bw
            bits = np.unpackbits(np.frombuffer(buf[r.i:r.i + nbytes], dtype=np.uint8), bitorder='little')
            r.i += nbytes
            if bw == 0: vals = np.zeros(n, dtype=np.int64)
            else: vals = bits[:n * bw].reshape(n, bw).dot(1 << np.arange(bw))
            out.append(vals)
        else:
            n = h >> 1; w = (bw + 7) // 8
            v = int.from_bytes(buf[r.i:r.i + w], 'little') if w else 0; r.i += w
            out.append(np.full(n, v, dtype=np.int64))
        got += len(out[-1])
    return np.concatenate(out)[:count]


DT = {1: '<i4', 2: '<i8', 4: '<f4', 5: '<f8'}


def plain(buf, ptype, n):
    if ptype in DT: return np.frombuffer(buf, dtype=DT[ptype], count=n)
    if ptype == 6:
        res, i = [], 0
        for _ in range(n):
            ln = struct.unpack_from('<I', buf, i)[0]; i += 4
            res.append(buf[i:i + ln].decode('utf-8', 'replace')); i += ln
        return np.array(res, dtype=object)
    raise ValueError(f"unsupported type {ptype}")


def read_column(data, chunk, optional):
    m = chunk[3]; ptype, codec, nvals = m[1], m[4], m[5]
    start = m.get(11) or m[9]
    r = R(data, start); dictionary = None; parts = []; read = 0
    while read < nvals:
        h = r.struct(); body_start = r.i
        raw = data[body_start:body_start + h[3]]; r.i = body_start + h[3]
        if codec == 1: raw = snappy(raw)
        elif codec != 0: raise ValueError(f"unsupported codec {codec}")
        if h[1] == 2:  # dictionary page
            dictionary = plain(raw, ptype, h[7][1]); continue
        if h[1] != 0: raise ValueError(f"unsupported page type {h[1]}")
        dh = h[5]; n, enc = dh[1], dh[2]; pos = 0
        if optional:
            ln = struct.unpack_from('<I', raw, 0)[0]
            defs = rle_hybrid(raw[4:4 + ln], 1, n); pos = 4 + ln
        else:
            defs = np.ones(n, dtype=np.int64)
        nn = int(defs.sum()); body = raw[pos:]
        if enc in (2, 8):
            bw = body[0]; idx = rle_hybrid(body[1:], bw, nn); vals = dictionary[idx]
        elif enc == 0:
            vals = plain(body, ptype, nn)
        else:
            raise ValueError(f"unsupported encoding {enc}")
        if nn == n: parts.append(vals)
        else:
            full = np.full(n, np.nan, dtype=object if vals.dtype == object else float)
            full[defs == 1] = vals; parts.append(full)
        read += n
    return np.concatenate(parts)


def read_parquet(path):
    data = open(path, 'rb').read()
    assert data[:4] == b'PAR1' and data[-4:] == b'PAR1'
    flen = struct.unpack('<I', data[-8:-4])[0]
    meta = R(data[-8 - flen:-8]).struct()
    schema = meta[2][1:]; cols = {}
    for rg in meta[4]:
        for el, chunk in zip(schema, rg[1]):
            name = el[4].decode(); optional = el.get(3) == 1
            cols.setdefault(name, []).append(read_column(data, chunk, optional))
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    assert len(df) == meta[3], (len(df), meta[3])
    return df


if __name__ == '__main__':
    df = read_parquet(sys.argv[1]); df.to_csv(sys.argv[2], index=False)
    print(sys.argv[1], '->', sys.argv[2], df.shape)
