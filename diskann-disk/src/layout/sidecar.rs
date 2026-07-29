/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::data_model::{GraphHeader, LayoutFingerprint};

const MAGIC: &[u8; 8] = b"DNLYT001";
const FORMAT_VERSION: u32 = 1;
const FIXED_SIZE: u64 = 160;

/// Errors produced while rewriting or loading graph-aware layouts.
#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid graph-aware layout: {0}")]
    Invalid(String),
    #[error("refusing to overwrite existing path {0}")]
    AlreadyExists(PathBuf),
}

pub(crate) type Result<T> = std::result::Result<T, LayoutError>;

pub(crate) fn io_error(path: &Path, source: std::io::Error) -> LayoutError {
    LayoutError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Immutable logical-to-physical slot mapping.
#[derive(Debug, Clone)]
pub struct PhysicalLayout {
    logical_to_physical: Arc<[u32]>,
    fingerprint: LayoutFingerprint,
}

impl PhysicalLayout {
    /// Return the physical record slot for a logical node ID.
    #[inline]
    pub fn physical_slot(&self, logical_id: u32) -> Option<u32> {
        self.logical_to_physical.get(logical_id as usize).copied()
    }

    pub fn len(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn is_empty(&self) -> bool {
        self.logical_to_physical.is_empty()
    }

    /// Heap bytes occupied by the mapping.
    pub fn memory_bytes(&self) -> usize {
        self.logical_to_physical.len() * size_of::<u32>()
    }

    pub const fn fingerprint(&self) -> LayoutFingerprint {
        self.fingerprint
    }

    pub fn mapping(&self) -> &[u32] {
        &self.logical_to_physical
    }

    #[cfg(test)]
    pub(crate) fn from_mapping_for_test(mapping: Vec<u32>) -> Arc<Self> {
        Arc::new(Self {
            logical_to_physical: mapping.into(),
            fingerprint: LayoutFingerprint::new(1).expect("non-zero test fingerprint"),
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StructuralFields {
    pub num_points: u64,
    pub dims: u64,
    pub medoid: u64,
    pub node_len: u64,
    pub block_size: u64,
    pub nodes_per_block: u64,
    pub associated_data_length: u64,
    pub index_len: u64,
}

impl StructuralFields {
    pub(crate) fn from_header(header: &GraphHeader, index_len: u64) -> Result<Self> {
        let metadata = header.metadata();
        let associated_data_length = u64::try_from(metadata.associated_data_length)
            .map_err(|_| LayoutError::Invalid("associated-data length exceeds u64".into()))?;
        Ok(Self {
            num_points: metadata.num_pts,
            dims: metadata.dims as u64,
            medoid: metadata.medoid,
            node_len: metadata.node_len,
            block_size: header.block_size(),
            nodes_per_block: metadata.num_nodes_per_block,
            associated_data_length,
            index_len,
        })
    }

    pub(crate) fn update_digest(&self, digest: &mut Sha256) {
        for value in [
            self.num_points,
            self.dims,
            self.medoid,
            self.node_len,
            self.block_size,
            self.nodes_per_block,
            self.associated_data_length,
            self.index_len,
        ] {
            digest.update(value.to_le_bytes());
        }
    }
}

/// Return `<index>.layout`.
pub fn default_sidecar_path(index_path: impl AsRef<Path>) -> PathBuf {
    let mut value = index_path.as_ref().as_os_str().to_os_string();
    value.push(".layout");
    PathBuf::from(value)
}

pub(crate) fn read_index_header(path: &Path) -> Result<(GraphHeader, u64)> {
    let index_len = fs::metadata(path)
        .map_err(|error| io_error(path, error))?
        .len();
    let required = 8 + GraphHeader::get_size();
    if index_len < required as u64 {
        return Err(LayoutError::Invalid(format!(
            "index {} is shorter than its {}-byte header",
            path.display(),
            required
        )));
    }
    let mut bytes = vec![0; required];
    File::open(path)
        .map_err(|error| io_error(path, error))?
        .read_exact(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    let header = GraphHeader::try_from(&bytes[8..])
        .map_err(|error| LayoutError::Invalid(format!("cannot parse graph header: {error}")))?;
    Ok((header, index_len))
}

pub(crate) fn expected_index_len(fields: &StructuralFields) -> Result<u64> {
    if fields.block_size == 0 || fields.node_len == 0 {
        return Err(LayoutError::Invalid(
            "block size and node length must be non-zero".into(),
        ));
    }
    let data_blocks = if fields.nodes_per_block > 0 {
        if fields.nodes_per_block != fields.block_size / fields.node_len {
            return Err(LayoutError::Invalid(format!(
                "nodes/block {} does not match block-size/node-length geometry",
                fields.nodes_per_block
            )));
        }
        fields.num_points.div_ceil(fields.nodes_per_block)
    } else {
        let blocks_per_node = fields.node_len.div_ceil(fields.block_size);
        fields
            .num_points
            .checked_mul(blocks_per_node)
            .ok_or_else(|| LayoutError::Invalid("index geometry overflows u64".into()))?
    };
    fields
        .block_size
        .checked_mul(data_blocks + 1)
        .ok_or_else(|| LayoutError::Invalid("index length overflows u64".into()))
}

pub(crate) fn mapping_bytes_digest(mapping: &[u32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for &slot in mapping {
        digest.update(slot.to_le_bytes());
    }
    digest.finalize().into()
}

pub(crate) fn binding_digest(
    source_digest: &[u8; 32],
    fields: &StructuralFields,
    mapping: &[u32],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"DiskANN graph-aware physical layout binding v1");
    digest.update(source_digest);
    fields.update_digest(&mut digest);
    for &slot in mapping {
        digest.update(slot.to_le_bytes());
    }
    digest.finalize().into()
}

pub(crate) fn digest_fingerprint(digest: &[u8; 32]) -> Result<LayoutFingerprint> {
    let prefix = u64::from_le_bytes(digest[..8].try_into().expect("eight-byte digest prefix"));
    LayoutFingerprint::new(prefix)
        .ok_or_else(|| LayoutError::Invalid("binding digest has a zero fingerprint prefix".into()))
}

pub(crate) fn validate_mapping(mapping: &[u32], num_points: u64) -> Result<()> {
    if num_points > u32::MAX as u64 {
        return Err(LayoutError::Invalid(
            "layout sidecars support at most u32::MAX points".into(),
        ));
    }
    if mapping.len() as u64 != num_points {
        return Err(LayoutError::Invalid(format!(
            "mapping contains {} entries, expected {num_points}",
            mapping.len()
        )));
    }
    let mut seen = vec![false; mapping.len()];
    for (logical, &physical) in mapping.iter().enumerate() {
        let physical = physical as usize;
        if physical >= mapping.len() {
            return Err(LayoutError::Invalid(format!(
                "physical slot {physical} for logical node {logical} is out of range"
            )));
        }
        if std::mem::replace(&mut seen[physical], true) {
            return Err(LayoutError::Invalid(format!(
                "physical slot {physical} occurs more than once"
            )));
        }
    }
    Ok(())
}

pub(crate) struct SidecarContents<'a> {
    pub fields: &'a StructuralFields,
    pub source_digest: &'a [u8; 32],
    pub binding_digest: &'a [u8; 32],
    pub fingerprint: LayoutFingerprint,
    pub mapping: &'a [u32],
}

pub(crate) fn write_sidecar(path: &Path, contents: SidecarContents<'_>) -> Result<()> {
    if path.exists() {
        return Err(LayoutError::AlreadyExists(path.to_path_buf()));
    }
    let file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| io_error(path, error))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(MAGIC)
        .and_then(|_| writer.write_all(&FORMAT_VERSION.to_le_bytes()))
        .and_then(|_| writer.write_all(&0u32.to_le_bytes()))
        .map_err(|error| io_error(path, error))?;
    for value in [
        contents.fields.num_points,
        contents.fields.node_len,
        contents.fields.block_size,
        contents.fields.nodes_per_block,
        contents.fields.index_len,
        contents.fingerprint.get(),
    ] {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| io_error(path, error))?;
    }
    writer
        .write_all(contents.source_digest)
        .and_then(|_| writer.write_all(contents.binding_digest))
        .and_then(|_| writer.write_all(&mapping_bytes_digest(contents.mapping)))
        .map_err(|error| io_error(path, error))?;
    for &slot in contents.mapping {
        writer
            .write_all(&slot.to_le_bytes())
            .map_err(|error| io_error(path, error))?;
    }
    writer.flush().map_err(|error| io_error(path, error))
}

fn read_u32(reader: &mut impl Read, path: &Path) -> Result<u32> {
    let mut bytes = [0; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read, path: &Path) -> Result<u64> {
    let mut bytes = [0; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    Ok(u64::from_le_bytes(bytes))
}

/// Load and strictly validate the sidecar for an index.
///
/// Legacy 0.0/1.0 indexes return `None` before allocating a mapping.
pub fn load_physical_layout(
    index_path: impl AsRef<Path>,
    sidecar_path: Option<&Path>,
) -> Result<Option<Arc<PhysicalLayout>>> {
    let index_path = index_path.as_ref();
    let (header, index_len) = read_index_header(index_path)?;
    if header.layout_version() != &GraphHeader::GRAPH_AWARE_LAYOUT_VERSION {
        if header.layout_version().major_version() == 0
            || header.layout_version() == &GraphHeader::CURRENT_LAYOUT_VERSION
        {
            return Ok(None);
        }
        return Err(LayoutError::Invalid(format!(
            "unsupported graph layout version {}",
            header.layout_version()
        )));
    }
    let fingerprint = header.metadata().layout_fingerprint().ok_or_else(|| {
        LayoutError::Invalid("graph layout 1.1 index has a zero fingerprint".into())
    })?;
    let fields = StructuralFields::from_header(&header, index_len)?;
    if fields.index_len != expected_index_len(&fields)?
        || header.metadata().disk_index_file_size != index_len
    {
        return Err(LayoutError::Invalid(
            "index file length does not match header geometry".into(),
        ));
    }

    let owned_path;
    let sidecar_path = match sidecar_path {
        Some(path) => path,
        None => {
            owned_path = default_sidecar_path(index_path);
            &owned_path
        }
    };
    let sidecar_len = fs::metadata(sidecar_path)
        .map_err(|error| io_error(sidecar_path, error))?
        .len();
    let expected_sidecar_len = FIXED_SIZE
        .checked_add(
            fields
                .num_points
                .checked_mul(4)
                .ok_or_else(|| LayoutError::Invalid("sidecar length overflows u64".into()))?,
        )
        .ok_or_else(|| LayoutError::Invalid("sidecar length overflows u64".into()))?;
    if sidecar_len != expected_sidecar_len {
        return Err(LayoutError::Invalid(format!(
            "sidecar length is {sidecar_len}, expected {expected_sidecar_len}"
        )));
    }

    let mut reader =
        BufReader::new(File::open(sidecar_path).map_err(|error| io_error(sidecar_path, error))?);
    let mut magic = [0; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|error| io_error(sidecar_path, error))?;
    if &magic != MAGIC
        || read_u32(&mut reader, sidecar_path)? != FORMAT_VERSION
        || read_u32(&mut reader, sidecar_path)? != 0
    {
        return Err(LayoutError::Invalid(
            "sidecar magic, version, or reserved field is invalid".into(),
        ));
    }
    let stored = [
        read_u64(&mut reader, sidecar_path)?,
        read_u64(&mut reader, sidecar_path)?,
        read_u64(&mut reader, sidecar_path)?,
        read_u64(&mut reader, sidecar_path)?,
        read_u64(&mut reader, sidecar_path)?,
        read_u64(&mut reader, sidecar_path)?,
    ];
    let expected = [
        fields.num_points,
        fields.node_len,
        fields.block_size,
        fields.nodes_per_block,
        fields.index_len,
        fingerprint.get(),
    ];
    if stored != expected {
        return Err(LayoutError::Invalid(
            "sidecar structural fields do not match the index header".into(),
        ));
    }
    let mut source_digest = [0; 32];
    let mut stored_binding = [0; 32];
    let mut stored_checksum = [0; 32];
    reader
        .read_exact(&mut source_digest)
        .and_then(|_| reader.read_exact(&mut stored_binding))
        .and_then(|_| reader.read_exact(&mut stored_checksum))
        .map_err(|error| io_error(sidecar_path, error))?;
    let mut mapping = Vec::with_capacity(fields.num_points as usize);
    for _ in 0..fields.num_points {
        mapping.push(read_u32(&mut reader, sidecar_path)?);
    }
    validate_mapping(&mapping, fields.num_points)?;
    if mapping_bytes_digest(&mapping) != stored_checksum {
        return Err(LayoutError::Invalid(
            "sidecar mapping checksum does not match".into(),
        ));
    }
    let computed_binding = binding_digest(&source_digest, &fields, &mapping);
    if computed_binding != stored_binding || digest_fingerprint(&computed_binding)? != fingerprint {
        return Err(LayoutError::Invalid(
            "sidecar binding digest does not match".into(),
        ));
    }

    Ok(Some(Arc::new(PhysicalLayout {
        logical_to_physical: mapping.into(),
        fingerprint,
    })))
}
