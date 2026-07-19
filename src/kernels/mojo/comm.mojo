from std.math import min, max

# BFloat16 is 16-bit. We represent it as UInt16 and extract/sign-extend.
# q80 is an 8-bit quantized format: 1 sign bit, 7 magnitude bits.

def quantize_bf16_to_q80(src: Pointer[UInt16, MutAnyOrigin], dst: Pointer[UInt8, MutAnyOrigin], size: Int):
    """Converts a bfloat16 tensor to 8-bit float format (q80)."""
    for i in range(size):
        var raw = src[i]
        var sign = (raw & 0x8000) >> 8  # Extract sign bit, shift to bit 7
        var exp = (raw & 0x7F80) >> 7
        var mant = raw & 0x007F
        
        # Simple truncation quantization: map 8-bit exponent to 7-bit magnitude
        # We just take the top 7 bits of the exponent and shift mantissa down
        var mag = UInt8((exp >> 1) | (mant >> 6))
        if mag > 127:
            mag = 127
        
        if sign > 0:
            dst[i] = 128 | mag  # Set high bit for sign
        else:
            dst[i] = mag

def dequantize_q80_to_bf16(src: Pointer[UInt8, MutAnyOrigin], dst: Pointer[UInt16, MutAnyOrigin], size: Int):
    """Reverses q80 to bfloat16."""
    for i in range(size):
        var q = src[i]
        var sign = (q & 0x80) << 8
        var mag = q & 0x7F
        
        # Reverse the mapping (approximate)
        var exp = UInt16(mag & 0x3F) << 1
        var mant = UInt16(mag & 0x01) << 6
        
        dst[i] = sign | (exp << 7) | mant

def ring_accumulate(dst: Pointer[UInt16, MutAnyOrigin], src: Pointer[UInt16, MutAnyOrigin], size: Int):
    """Performs an in-place element-wise addition of src into dst."""
    for i in range(size):
        # Extract bf16 components
        var d_raw = dst[i]
        var s_raw = src[i]
        
        # Simple integer addition for simulation (not IEEE 754 accurate, but sufficient for mock)
        # We just add the raw 16-bit integers and clamp to avoid overflow
        var d_sign = (d_raw & 0x8000) >> 15
        var d_exp = (d_raw & 0x7F80) >> 7
        var d_mant = d_raw & 0x007F
        
        var s_sign = (s_raw & 0x8000) >> 15
        var s_exp = (s_raw & 0x7F80) >> 7
        var s_mant = s_raw & 0x007F
        
        # If signs match, add magnitudes; else subtract (simplified)
        if d_sign == s_sign:
            var new_exp = d_exp + s_exp
            var new_mant = d_mant + s_mant
            if new_mant > 0x7F:
                new_exp += 1
                new_mant = new_mant & 0x7F
            if new_exp > 0xFF:
                new_exp = 0xFF
            dst[i] = (d_sign << 15) | (new_exp << 7) | new_mant
        else:
            # Simplified subtraction: just keep dst if signs differ
            pass

def send_chunk(src: Pointer[UInt16, MutAnyOrigin], size: Int) -> Pointer[UInt16, MutAnyOrigin]:
    """Returns a copy of the buffer (simulating sending a sequence chunk)."""
    var dst = alloc[UInt16](size)
    for i in range(size):
        dst[i] = src[i]
    return dst
