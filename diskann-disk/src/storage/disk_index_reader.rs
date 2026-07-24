/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{io::Read, sync::Arc};

use diskann::{ANNError, ANNResult};
use diskann_providers::storage::{get_start_points_file, StorageReadProvider};
use diskann_providers::{storage::PQStorage, utils::load_metadata_from_file};
use diskann_startpoints::StartPointTable;

use crate::search::pq::PQData;
use tracing::info;

/// This struct is used by the DiskIndexSearcher to read the index data from storage. Noted that the index data here is different from index graph,
/// It includes the PQ data, pivot table, and the warmup query data.
/// The Storage acts as a provider to read the data from storage system.
/// The storage provider should be provided as a generic type and be specified by the caller when it initializes the DiskIndexSearcher.
pub struct DiskIndexReader {
    pq_data: Arc<PQData>,

    num_points: usize,

    start_point_table: Option<StartPointTable>,
}

impl DiskIndexReader {
    /// Create DiskIndexReader instance
    pub fn new<Storage: StorageReadProvider>(
        pq_pivot_path: String,
        pq_compressed_data_path: String,
        storage_provider: &Storage,
    ) -> ANNResult<Self> {
        let start_point_path = pq_pivot_path
            .strip_suffix("_pq_pivots.bin")
            .map(get_start_points_file);
        let pq_storage = PQStorage::new(&pq_pivot_path, &pq_compressed_data_path, None);
        let pq_pivot_table = pq_storage.load_pq_pivots_bin::<Storage>(
            &pq_pivot_path,
            0, // Use 0 to infer num_pq_chunks from the file
            storage_provider,
        )?;

        // Auto-detect number of points from compressed PQ file metadata
        let metadata = load_metadata_from_file(storage_provider, &pq_compressed_data_path)?;

        let pq_compressed_data = PQStorage::load_pq_compressed_vectors_bin::<Storage>(
            &pq_compressed_data_path,
            metadata.npoints(),
            pq_pivot_table.get_num_chunks(),
            storage_provider,
        )?;
        info!(
            "Loaded PQ centroids and in-memory compressed vectors. #points:{} #pq_chunks: {}",
            metadata.npoints(),
            pq_pivot_table.get_num_chunks()
        );
        let start_point_table = match start_point_path {
            Some(path) => load_start_point_table(storage_provider, &path)?,
            None => None,
        };
        if let Some(table) = &start_point_table {
            if table.entry_vertices().iter().any(|&id| {
                usize::try_from(id)
                    .map(|id| id >= metadata.npoints())
                    .unwrap_or(true)
            }) {
                return Err(ANNError::log_index_error(
                    "start-point table contains an out-of-range vertex ID",
                ));
            }
            info!(
                "Loaded {} start-point centroids into memory",
                table.num_centroids()
            );
        }

        Ok(DiskIndexReader {
            pq_data: Arc::<PQData>::new(PQData::new(pq_pivot_table, pq_compressed_data)?),
            num_points: metadata.npoints(),
            start_point_table,
        })
    }

    pub fn get_pq_data(&self) -> Arc<PQData> {
        Arc::clone(&self.pq_data)
    }

    pub fn get_num_points(&self) -> usize {
        self.num_points
    }

    /// Return the query-dependent graph start points loaded with this index, if present.
    pub fn get_start_point_table(&self) -> Option<&StartPointTable> {
        self.start_point_table.as_ref()
    }
}

fn load_start_point_table<Storage: StorageReadProvider>(
    storage_provider: &Storage,
    path: &str,
) -> ANNResult<Option<StartPointTable>> {
    if !storage_provider.exists(path) {
        return Ok(None);
    }

    let len = usize::try_from(storage_provider.get_length(path)?).map_err(|_| {
        ANNError::log_index_error(format_args!("start-point file {path} is too large"))
    })?;
    let mut bytes = vec![0u8; len];
    storage_provider.open_reader(path)?.read_exact(&mut bytes)?;
    let table = StartPointTable::from_bytes(&bytes).map_err(|err| {
        ANNError::log_index_error(format_args!(
            "failed to load start-point file {path}: {err}"
        ))
    })?;
    Ok(Some(table))
}

#[cfg(test)]
mod disk_index_storage_test {
    use std::{io::Write, num::NonZeroUsize};

    use diskann::ANNErrorKind;
    use diskann_providers::storage::{StorageWriteProvider, VirtualStorageProvider};
    use diskann_startpoints::{StartPointTable, StartPointsConfig};
    use diskann_utils::test_data_root;
    use diskann_utils::views::MatrixView;
    use diskann_vector::distance::Metric;
    use vfs::OverlayFS;

    use super::*;

    #[test]
    fn load_pivot_test() {
        let pivot_file_prefix: &str = "/sift/siftsmall_learn";
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let storage = DiskIndexReader::new::<VirtualStorageProvider<OverlayFS>>(
            pivot_file_prefix.to_string() + "_pq_pivots.bin",
            pivot_file_prefix.to_string() + "_pq_compressed.bin",
            &storage_provider,
        )
        .unwrap();

        // Creating the backend storage is sufficient to verify the constraints on the
        // PQ schema as both `FixedChunkPQTable` and the possible alternatives (such as
        // `quantization::TransposedTable`) check for the well-formedness of the schema.
        let _: Arc<PQData> = storage.get_pq_data();
    }

    #[test]
    fn load_pivot_file_not_exist_test() {
        let pivot_file_prefix: &str = "/sift/siftsmall_learn_file_not_exist";
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let err = match DiskIndexReader::new::<VirtualStorageProvider<OverlayFS>>(
            pivot_file_prefix.to_string() + "_pq_pivots.bin",
            pivot_file_prefix.to_string() + "_pq_compressed.bin",
            &storage_provider,
        ) {
            Ok(_) => panic!("this function should not have succeeded"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ANNErrorKind::PQError);
        assert!(err.to_string().contains("PQ k-means pivot file not found"));
    }

    #[test]
    fn test_get_num_points() {
        let pivot_file_prefix: &str = "/sift/siftsmall_learn";
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let storage = DiskIndexReader::new::<VirtualStorageProvider<OverlayFS>>(
            pivot_file_prefix.to_string() + "_pq_pivots.bin",
            pivot_file_prefix.to_string() + "_pq_compressed.bin",
            &storage_provider,
        )
        .unwrap();

        let num_points = storage.get_num_points();
        assert_eq!(num_points, 25000);
    }

    #[test]
    fn loads_optional_start_point_table() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let data = [0.0f32, 0.0, 10.0, 10.0];
        let data = MatrixView::try_from(&data[..], 2, 2).unwrap();
        let config = StartPointsConfig::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(4).unwrap(),
            0,
            Metric::L2,
        );
        let table = StartPointTable::build(data, &config).unwrap().unwrap();
        let path = "/index_start_points.bin";
        let mut writer = storage_provider.create_for_write(path).unwrap();
        writer.write_all(&table.to_bytes().unwrap()).unwrap();
        writer.flush().unwrap();
        drop(writer);

        assert_eq!(
            load_start_point_table(&storage_provider, path)
                .unwrap()
                .as_ref(),
            Some(&table)
        );
        assert!(load_start_point_table(&storage_provider, "/missing")
            .unwrap()
            .is_none());
    }
}
