// GGUF file parser. Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//
// Layout of a GGUF file (all integers little-endian):
//   magic "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64
//   metadata KV pairs (key: string, type tag: u32, value)
//   tensor infos (name, n_dims u32, dims [u64], dtype u32, offset u64)
//   padding to `general.alignment` (default 32)
//   tensor data blob (tensor offsets are relative to the start of this blob)

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: usize = 32;

// ---------------------------------------------------------------------------
// Cursor: a bounds-checked reader over the mmapped bytes.
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .with_context(|| {
                format!("unexpected EOF: need {} bytes at offset {}", n, self.pos)
            })?;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// GGUF string: u64 byte length, then UTF-8 bytes. No null terminator.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8(bytes.to_vec()).context("string is not valid UTF-8")?)
    }
}

// ---------------------------------------------------------------------------
// Metadata values.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// `type_tag` values are fixed by the spec: 0=u8, 1=i8, ... 9=array, ...
    fn read(cur: &mut Cursor, type_tag: u32) -> Result<GgufValue> {
        Ok(match type_tag {
            0 => GgufValue::U8(cur.read_u8()?),
            1 => GgufValue::I8(cur.read_u8()? as i8),
            2 => GgufValue::U16(cur.read_u16()?),
            3 => GgufValue::I16(cur.read_u16()? as i16),
            4 => GgufValue::U32(cur.read_u32()?),
            5 => GgufValue::I32(cur.read_u32()? as i32),
            6 => GgufValue::F32(cur.read_f32()?),
            7 => GgufValue::Bool(cur.read_u8()? != 0),
            8 => GgufValue::String(cur.read_string()?),
            9 => {
                let elem_tag = cur.read_u32()?;
                let count = cur.read_u64()? as usize;
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(GgufValue::read(cur, elem_tag)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(cur.read_u64()?),
            11 => GgufValue::I64(cur.read_u64()? as i64),
            12 => GgufValue::F64(cur.read_f64()?),
            other => bail!("unknown metadata value type tag {}", other),
        })
    }

    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            GgufValue::U32(v) => Some(v),
            GgufValue::U64(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            GgufValue::U8(v) => Some(v as i64),
            GgufValue::I8(v) => Some(v as i64),
            GgufValue::U16(v) => Some(v as i64),
            GgufValue::I16(v) => Some(v as i64),
            GgufValue::U32(v) => Some(v as i64),
            GgufValue::I32(v) => Some(v as i64),
            GgufValue::U64(v) => i64::try_from(v).ok(),
            GgufValue::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(items) => Some(items),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor info.
// ---------------------------------------------------------------------------

/// ggml tensor data types. Discriminants are the on-disk u32 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Bf16 = 30,
}

impl GgmlType {
    /// (elements per block, bytes per block) for dtypes we can decode.
    /// Quantized formats not yet implemented return None.
    pub fn block_layout(self) -> Option<(usize, usize)> {
        match self {
            GgmlType::F32 => Some((1, 4)),
            GgmlType::F16 => Some((1, 2)),
            GgmlType::Q8_0 => Some((crate::quant::QK8_0, crate::quant::Q8_0_BLOCK_BYTES)),
            _ => None,
        }
    }

    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            30 => GgmlType::Bf16,
            other => bail!("unsupported ggml tensor type {}", other),
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in ggml order: dims[0] is the fastest-varying (contiguous)
    /// axis. This is the REVERSE of PyTorch shape order.
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Byte offset relative to the start of the tensor data blob.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// Size of this tensor's data on disk, derived from dtype block layout.
    pub fn byte_size(&self) -> Result<usize> {
        let (block_elems, block_bytes) = self
            .dtype
            .block_layout()
            .with_context(|| format!("dtype {:?} not supported yet", self.dtype))?;
        let n = self.n_elements();
        ensure!(
            n % block_elems == 0,
            "tensor '{}' has {} elements, not a multiple of block size {}",
            self.name,
            n,
            block_elems
        );
        Ok(n / block_elems * block_bytes)
    }

    fn read(cur: &mut Cursor) -> Result<TensorInfo> {
        let name = cur.read_string()?;
        let n_dims = cur.read_u32()? as usize;
        ensure!(n_dims <= 4, "tensor '{}' has {} dims, max is 4", name, n_dims);
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(cur.read_u64()?);
        }
        let dtype = GgmlType::from_u32(cur.read_u32()?)
            .with_context(|| format!("tensor '{}'", name))?;
        let offset = cur.read_u64()?;
        Ok(TensorInfo { name, dims, dtype, offset })
    }
}

// ---------------------------------------------------------------------------
// The file itself.
// ---------------------------------------------------------------------------

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the tensor data blob begins.
    pub data_start: usize,
}

/// Parsed header contents, separated from GgufFile so tests can run on plain
/// byte slices without a real file.
struct Parsed {
    version: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    data_start: usize,
}

fn parse(data: &[u8]) -> Result<Parsed> {
    let mut cur = Cursor::new(data);

    let magic = cur.take(4)?;
    ensure!(magic == GGUF_MAGIC, "not a GGUF file (bad magic {:02x?})", magic);
    let version = cur.read_u32()?;
    ensure!(
        version == 2 || version == 3,
        "unsupported GGUF version {} (only 2 and 3)",
        version
    );

    let tensor_count = cur.read_u64()? as usize;
    let kv_count = cur.read_u64()? as usize;

    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = cur.read_string()?;
        let type_tag = cur.read_u32()?;
        let value = GgufValue::read(&mut cur, type_tag)
            .with_context(|| format!("metadata key '{}'", key))?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        tensors.push(TensorInfo::read(&mut cur)?);
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_ALIGNMENT);
    ensure!(
        alignment.is_power_of_two(),
        "general.alignment {} is not a power of two",
        alignment
    );
    let data_start = cur.pos.next_multiple_of(alignment);
    ensure!(
        data_start <= data.len(),
        "file truncated: tensor data would start at {} but file is {} bytes",
        data_start,
        data.len()
    );

    Ok(Parsed { version, metadata, tensors, data_start })
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile> {
        let file = File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        // Safety: the map is read-only; we accept UB if another process
        // truncates the file while we hold the map.
        let mmap = unsafe { Mmap::map(&file)? };
        let parsed = parse(&mmap)?;
        Ok(GgufFile {
            mmap,
            version: parsed.version,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            data_start: parsed.data_start,
        })
    }

    pub fn meta_str(&self, key: &str) -> Result<&str> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_str())
            .with_context(|| format!("metadata key '{}' missing or not a string", key))
    }

    pub fn meta_array(&self, key: &str) -> Result<&[GgufValue]> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_array())
            .with_context(|| format!("metadata key '{}' missing or not an array", key))
    }

    /// Look up a tensor by name and return its info plus the raw byte slice
    /// of its data, straight out of the mmap (no copy).
    pub fn tensor(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .with_context(|| format!("no tensor named '{}'", name))?;
        let start = self.data_start + info.offset as usize;
        let size = info.byte_size()?;
        let end = start
            .checked_add(size)
            .filter(|&end| end <= self.mmap.len())
            .with_context(|| {
                format!(
                    "tensor '{}' data [{}, {}+{}) runs past end of file ({} bytes)",
                    info.name,
                    start,
                    start,
                    size,
                    self.mmap.len()
                )
            })?;
        Ok((info, &self.mmap[start..end]))
    }

    pub fn dump(&self, with_metadata: bool) {
        println!(
            "gguf version {} | {} metadata keys | {} tensors | data starts at byte {}",
            self.version,
            self.metadata.len(),
            self.tensors.len(),
            self.data_start
        );

        if with_metadata {
            println!("\nmetadata:");
            let mut keys: Vec<_> = self.metadata.keys().collect();
            keys.sort();
            for key in keys {
                println!("  {} = {}", key, format_value(&self.metadata[key]));
            }
        }

        println!("\ntensors (dims in ggml order, fastest-varying first):");
        for (i, t) in self.tensors.iter().enumerate() {
            let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
            println!(
                "  {:4}  {:40}  [{}]  {:?}  offset {}",
                i,
                t.name,
                dims.join(", "),
                t.dtype,
                t.offset
            );
        }
    }
}

/// Render a metadata value, truncating huge arrays (the vocab is 150k strings).
fn format_value(v: &GgufValue) -> String {
    match v {
        GgufValue::String(s) => {
            let mut s = s.as_str();
            if s.len() > 80 {
                let mut end = 80;
                while !s.is_char_boundary(end) {
                    end -= 1;
                }
                s = &s[..end];
            }
            format!("{:?}", s)
        }
        GgufValue::Array(items) => {
            let shown: Vec<String> = items.iter().take(8).map(format_value).collect();
            let ellipsis = if items.len() > 8 { ", ..." } else { "" };
            format!("[{}{}] ({} items)", shown.join(", "), ellipsis, items.len())
        }
        GgufValue::U8(v) => v.to_string(),
        GgufValue::I8(v) => v.to_string(),
        GgufValue::U16(v) => v.to_string(),
        GgufValue::I16(v) => v.to_string(),
        GgufValue::U32(v) => v.to_string(),
        GgufValue::I32(v) => v.to_string(),
        GgufValue::F32(v) => v.to_string(),
        GgufValue::Bool(v) => v.to_string(),
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F64(v) => v.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests: build a minimal GGUF file byte-by-byte and parse it back.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct Builder(Vec<u8>);

    impl Builder {
        fn u32(&mut self, v: u32) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(&mut self, v: u64) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn f32(&mut self, v: f32) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn string(&mut self, s: &str) -> &mut Self {
            self.u64(s.len() as u64);
            self.0.extend_from_slice(s.as_bytes());
            self
        }
    }

    /// A GGUF v3 file with 1 tensor and 3 metadata keys, hand-assembled.
    fn synthetic_file() -> Vec<u8> {
        let mut b = Builder(Vec::new());
        b.0.extend_from_slice(b"GGUF");
        b.u32(3); // version
        b.u64(1); // tensor count
        b.u64(3); // metadata kv count

        // general.architecture = "qwen2" (string, tag 8)
        b.string("general.architecture").u32(8).string("qwen2");
        // qwen2.rope.freq_base = 1000000.0 (f32, tag 6)
        b.string("qwen2.rope.freq_base").u32(6).f32(1_000_000.0);
        // tokenizer.ggml.tokens = ["a", "b"] (array of string: tag 9, elem 8)
        b.string("tokenizer.ggml.tokens")
            .u32(9)
            .u32(8)
            .u64(2)
            .string("a")
            .string("b");

        // tensor info: "blk.0.attn_q.weight", dims [896, 896], Q8_0, offset 0
        b.string("blk.0.attn_q.weight");
        b.u32(2).u64(896).u64(896);
        b.u32(8); // Q8_0
        b.u64(0);

        // pad to 32 and add a dummy data blob
        while b.0.len() % 32 != 0 {
            b.0.push(0);
        }
        b.0.extend_from_slice(&[0xAB; 64]);
        b.0
    }

    #[test]
    fn parses_synthetic_file() {
        let bytes = synthetic_file();
        let p = parse(&bytes).unwrap();

        assert_eq!(p.version, 3);
        assert_eq!(p.metadata.len(), 3);
        assert_eq!(
            p.metadata["general.architecture"],
            GgufValue::String("qwen2".into())
        );
        assert_eq!(p.metadata["qwen2.rope.freq_base"], GgufValue::F32(1_000_000.0));
        assert_eq!(
            p.metadata["tokenizer.ggml.tokens"],
            GgufValue::Array(vec![
                GgufValue::String("a".into()),
                GgufValue::String("b".into()),
            ])
        );

        assert_eq!(p.tensors.len(), 1);
        let t = &p.tensors[0];
        assert_eq!(t.name, "blk.0.attn_q.weight");
        assert_eq!(t.dims, vec![896, 896]);
        assert_eq!(t.dtype, GgmlType::Q8_0);
        assert_eq!(t.offset, 0);

        // header ends before the pad, data_start lands on the 32 boundary
        assert_eq!(p.data_start % 32, 0);
        assert_eq!(bytes[p.data_start], 0xAB);
    }

    #[test]
    fn rejects_bad_magic() {
        let err = parse(b"GGML\x03\x00\x00\x00").err().unwrap();
        assert!(err.to_string().contains("bad magic"));
    }

    #[test]
    fn rejects_truncated_file() {
        let mut bytes = synthetic_file();
        bytes.truncate(40); // mid-metadata
        assert!(parse(&bytes).is_err());
    }
}
