/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Cursor, num::NonZeroU64};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::{ANNError, ANNResult};

/// Non-zero binding between a graph-aware index and its physical-layout sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayoutFingerprint(NonZeroU64);

impl LayoutFingerprint {
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Index graph metadata. The metadata is stored in the first sector of the disk index file, or the first segment of the BigStorageStream.
/// The metadata is like a "header" of the index graph.
#[derive(Debug, Clone)]
pub struct GraphMetadata {
    // Number of points.
    pub num_pts: u64,

    // Data dimension.
    pub dims: usize,

    // Medoid index.
    pub medoid: u64,

    // Node length.
    pub node_len: u64,

    // Number of nodes per sector.
    pub num_nodes_per_block: u64,

    // Number of frozen nodes in Vamana index.
    pub vamana_frozen_num: u64,

    // Location of frozen nodes in Vamana index.
    pub vamana_frozen_loc: u64,

    // Binding to the physical-layout sidecar. Legacy layouts serialize this as zero.
    layout_fingerprint: Option<LayoutFingerprint>,

    // Size of the disk index file.
    pub disk_index_file_size: u64,

    // Length of the associated data.
    pub associated_data_length: usize,
}

#[allow(clippy::too_many_arguments)]
impl GraphMetadata {
    /// Create a new `GraphMetadata` object.
    pub fn new(
        num_pts: u64,
        dims: usize,
        medoid: u64,
        node_len: u64,
        num_nodes_per_sector: u64,
        vamana_frozen_num: u64,
        vamana_frozen_loc: u64,
        disk_index_file_size: u64,
        data_size: usize,
    ) -> Self {
        Self {
            num_pts,
            dims,
            medoid,
            node_len,
            num_nodes_per_block: num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            layout_fingerprint: None,
            disk_index_file_size,
            associated_data_length: data_size,
        }
    }

    /// Return the graph-aware physical-layout sidecar fingerprint, if present.
    pub const fn layout_fingerprint(&self) -> Option<LayoutFingerprint> {
        self.layout_fingerprint
    }

    /// Set the graph-aware physical-layout sidecar fingerprint.
    pub fn set_layout_fingerprint(&mut self, fingerprint: LayoutFingerprint) {
        self.layout_fingerprint = Some(fingerprint);
    }

    /// Serialize the `GraphMetadata` object to a byte vector.
    ///
    /// Layout:
    /// |number_of_points (8 bytes)| dimensions (8 bytes) | medoid (8 bytes) | node_len (8 bytes) | num_nodes_per_sector (8 bytes) | vamana_frozen_point_num (8 bytes) |
    /// ...| vamana_frozen_loc (8 bytes) | append_reorder_data (8 bytes) | disk_index_file_size (8 bytes) | associated_data_length (8 bytes) |
    /// `append_reorder_data` is zero for legacy layouts and is the layout fingerprint in 1.1.
    pub fn to_bytes(&self) -> ANNResult<Vec<u8>> {
        let mut buffer = vec![];

        let mut cursor = Cursor::new(&mut buffer);
        cursor.write_u64::<LittleEndian>(self.num_pts)?;
        cursor.write_u64::<LittleEndian>(self.dims as u64)?;
        cursor.write_u64::<LittleEndian>(self.medoid)?;
        cursor.write_u64::<LittleEndian>(self.node_len)?;
        cursor.write_u64::<LittleEndian>(self.num_nodes_per_block)?;
        cursor.write_u64::<LittleEndian>(self.vamana_frozen_num)?;
        cursor.write_u64::<LittleEndian>(self.vamana_frozen_loc)?;

        cursor
            .write_u64::<LittleEndian>(self.layout_fingerprint.map_or(0, LayoutFingerprint::get))?;
        cursor.write_u64::<LittleEndian>(self.disk_index_file_size)?;

        cursor.write_u64::<LittleEndian>(self.associated_data_length as u64)?;
        Ok(buffer)
    }

    // Size of the metadata after serialization.
    #[inline]
    pub fn get_size() -> usize {
        std::mem::size_of::<u64>() * 10
    }
}

impl<'a> TryFrom<&'a [u8]> for GraphMetadata {
    type Error = ANNError;
    /// Try creating a new `GraphMetadata` object from a byte slice. The try_from syntax is used here instead of from because this operation can fail.
    ///
    /// Layout:
    /// |number_of_points (8 bytes)| dimensions (8 bytes) | medoid (8 bytes) | node_len (8 bytes) | num_nodes_per_sector (8 bytes) | vamana_frozen_point_num (8 bytes) |
    /// ...| vamana_frozen_loc (8 bytes) | append_reorder_data (8 bytes) | disk_index_file_size (8 bytes) | associated_data_length (8 bytes) |
    fn try_from(value: &'a [u8]) -> ANNResult<Self> {
        if value.len() < Self::get_size() {
            return Err(ANNError::log_parse_slice_error(
                "&[u8]".to_string(),
                "GraphMetadata".to_string(),
                "The given bytes are not long enough to create a valid graph metadata.".to_string(),
            ));
        }

        let mut cursor = Cursor::new(value);
        let num_pts = cursor.read_u64::<LittleEndian>()?;
        let dims = cursor.read_u64::<LittleEndian>()?;
        let medoid = cursor.read_u64::<LittleEndian>()?;
        let node_len = cursor.read_u64::<LittleEndian>()?;
        let num_nodes_per_sector = cursor.read_u64::<LittleEndian>()?;
        let vamana_frozen_num = cursor.read_u64::<LittleEndian>()?;
        let vamana_frozen_loc = cursor.read_u64::<LittleEndian>()?;

        let layout_fingerprint = LayoutFingerprint::new(cursor.read_u64::<LittleEndian>()?);
        let disk_index_file_size = cursor.read_u64::<LittleEndian>()?;
        let associated_data_length = cursor.read_u64::<LittleEndian>()? as usize;

        Ok(Self {
            num_pts,
            dims: dims as usize,
            medoid,
            node_len,
            num_nodes_per_block: num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            layout_fingerprint,
            disk_index_file_size,
            associated_data_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphMetadata, LayoutFingerprint};

    #[test]
    fn test_graph_metadata_serialized_size() {
        let num_pts = 1000;
        let dims = 32;
        let medoid = 500;
        let node_len = 64;
        let num_nodes_per_sector = 4;
        let vamana_frozen_num = 20;
        let vamana_frozen_loc = 50;
        let disk_index_file_size = 1024;
        let data_size = 256;

        let metadata = GraphMetadata::new(
            num_pts,
            dims,
            medoid,
            node_len,
            num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            disk_index_file_size,
            data_size,
        );

        let bytes = metadata.to_bytes().unwrap();
        assert_eq!(bytes.len(), GraphMetadata::get_size());
    }

    #[test]
    fn test_graph_metadata_to_bytes_and_try_from() {
        let num_pts = 1000;
        let dims = 32;
        let medoid = 500;
        let node_len = 64;
        let num_nodes_per_sector = 4;
        let vamana_frozen_num = 20;
        let vamana_frozen_loc = 50;
        let disk_index_file_size = 1024;
        let data_size = 256;

        let metadata = GraphMetadata::new(
            num_pts,
            dims,
            medoid,
            node_len,
            num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            disk_index_file_size,
            data_size,
        );

        let bytes = metadata.to_bytes().unwrap();
        let deserialized_metadata = GraphMetadata::try_from(bytes.as_slice()).unwrap();
        assert_eq!(metadata.num_pts, deserialized_metadata.num_pts);
        assert_eq!(metadata.dims, deserialized_metadata.dims);
        assert_eq!(metadata.medoid, deserialized_metadata.medoid);
        assert_eq!(metadata.node_len, deserialized_metadata.node_len);
        assert_eq!(
            metadata.num_nodes_per_block,
            deserialized_metadata.num_nodes_per_block
        );
        assert_eq!(
            metadata.vamana_frozen_num,
            deserialized_metadata.vamana_frozen_num
        );
        assert_eq!(
            metadata.vamana_frozen_loc,
            deserialized_metadata.vamana_frozen_loc
        );
        assert_eq!(
            metadata.disk_index_file_size,
            deserialized_metadata.disk_index_file_size
        );
        assert_eq!(
            metadata.associated_data_length,
            deserialized_metadata.associated_data_length
        );
        assert_eq!(
            metadata.layout_fingerprint(),
            deserialized_metadata.layout_fingerprint()
        );
    }

    #[test]
    fn graph_metadata_round_trips_layout_fingerprint() {
        let mut metadata = GraphMetadata::new(1, 1, 0, 16, 1, 0, 0, 32, 0);
        metadata.set_layout_fingerprint(LayoutFingerprint::new(42).unwrap());

        let bytes = metadata.to_bytes().unwrap();
        assert_eq!(
            GraphMetadata::try_from(bytes.as_slice())
                .unwrap()
                .layout_fingerprint()
                .unwrap()
                .get(),
            42
        );
    }

    #[test]
    fn test_graph_metadata_try_from_error() {
        let bytes = vec![2; GraphMetadata::get_size() - 1];
        let result = GraphMetadata::try_from(&bytes[..]);
        assert!(result.is_err());
    }
}
