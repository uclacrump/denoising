import struct
import pdb

filename = "amide_export"

with open(filename, "rb") as f:
    while True:
        data = f.read(4)  # read 4 bytes
        if not data:
            break
        float_val = struct.unpack('<f', data)[0]
        pdb.set_trace()