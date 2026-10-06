//! dllama-net: cluster wire protocol, byte-compatible with the C++ engine (v0).
//!
//! Facts ported from the C++ source (see ../DESIGN.md §6 for the full table):
//! - control packets are raw `{ u32 position; u32 batch_size }` structs,
//!   `batch_size == 0` is the stop signal (`app.cpp:180-228`),
//! - weight chunks: `u32 nameSize (incl. NUL) | name | u32 opIndex | u64 offset
//!   | u64 nBytes | payload` (`nn-network.cpp:835-843`),
//! - the weight stream ends with a `u32 0` nameSize (`nn-network.cpp:814-819`).
//!
//! All integers little-endian (matches native-endian raw writes on every
//! platform the C++ engine supports).

#![allow(dead_code)]

use std::fmt;

pub const CONTROL_PACKET_BYTES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    UnexpectedEof,
    InvalidUtf8,
    StringTooLong,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireError::UnexpectedEof => write!(f, "unexpected end of buffer"),
            WireError::InvalidUtf8 => write!(f, "invalid utf-8 in string"),
            WireError::StringTooLong => write!(f, "string length exceeds u32 range"),
        }
    }
}
impl std::error::Error for WireError {}

pub type Result<T> = std::result::Result<T, WireError>;

/// Root -> workers, once per forward step (`LlmControlPacket`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlPacket {
    pub position: u32,
    pub batch_size: u32,
}

impl ControlPacket {
    /// `batch_size == 0` tells workers to exit their loop.
    pub const STOP: ControlPacket = ControlPacket { position: 0, batch_size: 0 };

    pub const fn new(position: u32, batch_size: u32) -> Self {
        Self { position, batch_size }
    }

    pub const fn is_stop(&self) -> bool {
        self.batch_size == 0
    }

    pub fn encode(&self) -> [u8; CONTROL_PACKET_BYTES] {
        let mut b = [0u8; CONTROL_PACKET_BYTES];
        b[..4].copy_from_slice(&self.position.to_le_bytes());
        b[4..].copy_from_slice(&self.batch_size.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self> {
        if b.len() < CONTROL_PACKET_BYTES {
            return Err(WireError::UnexpectedEof);
        }
        let mut pos = [0u8; 4];
        let mut bs = [0u8; 4];
        pos.copy_from_slice(&b[..4]);
        bs.copy_from_slice(&b[4..8]);
        Ok(Self {
            position: u32::from_le_bytes(pos),
            batch_size: u32::from_le_bytes(bs),
        })
    }
}

/// One chunk of a streamed weight tensor (root -> worker).
///
/// Layout: `u32 nameSize` (bytes incl. NUL) | name bytes + NUL | `u32 opIndex`
/// | `u64 offset` | `u64 nBytes`, payload follows the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightChunkHeader {
    pub name: String,
    pub op_index: u32,
    pub offset: u64,
    pub n_bytes: u64,
}

/// `nameSize == 0` marks the end of the weight stream (`nn-network.cpp:814`).
pub const WEIGHT_STREAM_END: u32 = 0;

impl WeightChunkHeader {
    pub const fn header_len(name_size: u32) -> usize {
        4 + name_size as usize + 4 + 8 + 8
    }

    pub fn encode_into(&self, buf: &mut Vec<u8>) {
        let name_size = self.name.len() + 1; // incl. NUL, matches strlen+1
        buf.extend_from_slice(&(name_size as u32).to_le_bytes());
        buf.extend_from_slice(self.name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&self.op_index.to_le_bytes());
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.n_bytes.to_le_bytes());
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(Self::header_len(self.name.len() as u32 + 1));
        self.encode_into(&mut v);
        v
    }

    /// Decode from the front of `buf`; returns the header + number of bytes
    /// consumed (payload starts at that offset, `n_bytes` long).
    pub fn decode(buf: &[u8]) -> Result<(Self, usize)> {
        if buf.len() < 4 {
            return Err(WireError::UnexpectedEof);
        }
        let name_size = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        if name_size == 0 {
            return Err(WireError::StringTooLong); // end marker — caller checks first
        }
        let need = Self::header_len(name_size as u32);
        if buf.len() < need {
            return Err(WireError::UnexpectedEof);
        }
        let name_bytes = &buf[4..4 + name_size];
        if name_bytes.last() != Some(&0) {
            return Err(WireError::InvalidUtf8); // must be NUL-terminated
        }
        let name = std::str::from_utf8(&name_bytes[..name_size - 1])
            .map_err(|_| WireError::InvalidUtf8)?
            .to_string();
        let mut u32b = [0u8; 4];
        u32b.copy_from_slice(&buf[4 + name_size..8 + name_size]);
        let op_index = u32::from_le_bytes(u32b);
        let mut u64b = [0u8; 8];
        u64b.copy_from_slice(&buf[8 + name_size..16 + name_size]);
        let offset = u64::from_le_bytes(u64b);
        u64b.copy_from_slice(&buf[16 + name_size..24 + name_size]);
        let n_bytes = u64::from_le_bytes(u64b);
        Ok((Self { name, op_index, offset, n_bytes }, need))
    }
}

/// Encode a config-stream string (`u32 length incl. NUL | bytes | NUL`).
///
/// NOTE(open): the exact `writeString` encoding must be verified against
/// `src/nn/nn-network.cpp` before Phase 3 wire parity is claimed (DESIGN.md §11.4).
pub fn encode_wire_string(s: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + s.len() + 1);
    v.extend_from_slice(&((s.len() + 1) as u32).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
    v.push(0);
    v
}

pub fn decode_wire_string(buf: &[u8]) -> Result<(String, usize)> {
    if buf.len() < 4 {
        return Err(WireError::UnexpectedEof);
    }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if len == 0 || buf.len() < 4 + len {
        return Err(WireError::UnexpectedEof);
    }
    let bytes = &buf[4..4 + len];
    if bytes.last() != Some(&0) {
        return Err(WireError::InvalidUtf8);
    }
    let s = std::str::from_utf8(&bytes[..len - 1]).map_err(|_| WireError::InvalidUtf8)?;
    Ok((s.to_string(), 4 + len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_packet_layout() {
        let p = ControlPacket::new(1234, 8);
        let b = p.encode();
        assert_eq!(b, [0xd2, 0x04, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00]);
        assert_eq!(ControlPacket::decode(&b).unwrap(), p);
        assert!(!p.is_stop());
        assert!(ControlPacket::STOP.is_stop());
        let s = ControlPacket::STOP.encode();
        assert_eq!(ControlPacket::decode(&s).unwrap().is_stop(), true);
        assert!(ControlPacket::decode(&b[..7]).is_err());
    }

    #[test]
    fn weight_chunk_roundtrip() {
        let h = WeightChunkHeader {
            name: "blk.0.attn_q".into(),
            op_index: 2,
            offset: 4096,
            n_bytes: 1 << 20,
        };
        let buf = h.encode();
        let (back, consumed) = WeightChunkHeader::decode(&buf).unwrap();
        assert_eq!(back, h);
        assert_eq!(consumed, buf.len());
        // truncated -> error
        assert!(WeightChunkHeader::decode(&buf[..buf.len() - 1]).is_err());
    }

    #[test]
    fn weight_stream_end_marker() {
        // end of stream: u32 0
        let end = (WEIGHT_STREAM_END as u32).to_le_bytes();
        assert_eq!(end, [0, 0, 0, 0]);
        assert!(WeightChunkHeader::decode(&end).is_err()); // caller must check nameSize==0 first
    }

    #[test]
    fn wire_string_roundtrip() {
        let b = encode_wire_string("token_embd");
        assert_eq!(b[..4], [11, 0, 0, 0]); // len incl. NUL
        let (s, used) = decode_wire_string(&b).unwrap();
        assert_eq!(s, "token_embd");
        assert_eq!(used, b.len());
        assert!(decode_wire_string(&b[..b.len() - 1]).is_err());
    }
}
