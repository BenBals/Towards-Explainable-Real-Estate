//! This module contains the implementation of a Partion.
use algorithms::sweepline;
use common::{BpResult, CostFunction, Immo};
use derive_more::{From, Into};
use mongodb::bson::{oid::ObjectId, Document};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{
    borrow::Borrow,
    collections::HashMap,
    fmt::{Debug, Formatter, Result},
    fs,
    io::Write,
    ops::{Index, IndexMut},
};
use typed_index_collections::TiVec;

/// This struct is used to reference BlockEntries.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash)]
pub struct BlockEntryIdx(usize);

/// This struct is used to reference Blocks.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash)]
pub struct BlockIdx(usize);

/// This struct represents an Immo, with metadata required to do the partitioning efficiently.
#[derive(Clone, Debug)]
pub struct BlockEntry<I>
where
    I: Borrow<Immo>,
{
    immo: I,
    reachable: Vec<BlockEntryIdx>,
    block: BlockIdx,
}

/// This struct represents a Collection of BlockEntries in a Partition.
/// By our definition the entries in this Block should be pairwise reachable.
#[derive(Clone, Debug)]
pub struct Block {
    entries: Vec<BlockEntryIdx>,
}

/// This struct represents a partition, which consists of many blocks.
pub struct Partition<I: Borrow<Immo>, C> {
    blocks: TiVec<BlockIdx, Block>,
    cost_function: C,
    entries: TiVec<BlockEntryIdx, BlockEntry<I>>,
    epsilon: f64,
}

impl<I: Borrow<Immo>> BlockEntry<I> {
    fn new(immo: I, block: BlockIdx) -> BpResult<Self> {
        immo.borrow()
            .plane_location
            .ok_or("immo does not have plane location")?;
        Ok(Self {
            immo,
            block,
            reachable: Vec::new(),
        })
    }

    /// Calculates the distance to another Immo.
    /// For this [Immo::plane_distance](Immo::plane_distance) is used.
    pub fn distance(&self, other: &BlockEntry<I>) -> f64 {
        self.immo()
            .plane_distance(other.immo())
            // This will never be called as we know every referenced Immo has plane_distance
            .expect("could not compute distance")
    }

    /// Returns an Iterator over the indices of all directly reachable BlockEntries in the partition of this BlockEntry.
    /// This does not include self.
    pub fn iter_reachable(&self) -> impl Iterator<Item = BlockEntryIdx> + '_ {
        self.reachable.iter().copied()
    }

    /// Returns the block in which this BlockEntry currently resides.
    pub fn block(&self) -> BlockIdx {
        self.block
    }

    /// Returns a reference to its Immo
    pub fn immo(&self) -> &Immo {
        self.immo.borrow()
    }
}

impl Block {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Returns an Iterator over the indices of all BlockEntries in this Block.
    pub fn iter_entries(&self) -> impl Iterator<Item = BlockEntryIdx> + '_ {
        self.entries.iter().copied()
    }

    /// Same as [iter_entries](Block::iter_entries), but the iterator is rayon enabled.
    pub fn iter_par_entries(&self) -> impl ParallelIterator<Item = BlockEntryIdx> + '_ {
        self.entries.par_iter().copied()
    }

    /// Returns true if the block has no entries, else false
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Serialize, Deserialize)]
struct JsonSchema {
    partition: Vec<Vec<String>>,
    epsilon: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    collection: Option<String>,
}

impl<I: Borrow<Immo>, C: CostFunction> Partition<I, C> {
    /// Creates a new empty block in the current partition.
    /// # Returns
    /// Returns the BlockIdx of the new Block.
    pub fn create_new_block(&mut self) -> BlockIdx {
        self.blocks.push_and_get_key(Block::new())
    }

    /// Creates a new TiVec indexed by BlockIdx, where every value is set to start.
    pub fn create_block_data_vec<T: Clone>(&self, start: &T) -> TiVec<BlockIdx, T> {
        self.blocks.iter().map(|_| start.clone()).collect()
    }

    /// Creates a new TiVec indexed by BlockIdx, where every value is set to start.
    pub fn create_entry_data_vec<T: Clone>(&self, start: &T) -> TiVec<BlockEntryIdx, T> {
        self.entries.iter().map(|_| start.clone()).collect()
    }

    /// Returns an Iterator over the indices of all non-empty blocks in this partition.
    pub fn iter_blocks(&self) -> impl Iterator<Item = BlockIdx> + '_ {
        self.blocks.iter_enumerated().filter_map(|(idx, block)| {
            if block.entries.is_empty() {
                None
            } else {
                Some(idx)
            }
        })
    }

    /// Returns an Iterator over the indices of all entries.
    pub fn iter_entries(&self) -> impl Iterator<Item = BlockEntryIdx> + '_ {
        self.entries.keys()
    }

    /// Returns an Iterator over all immos in this partition.
    pub fn iter_immos(&self) -> impl Iterator<Item = &Immo> + '_ {
        self.iter_entries()
            .map(move |entry_idx| self[entry_idx].immo())
    }

    /// Creates a new BlockEntry in the block indexed by `block_idx`.
    /// This also creates the reachability edges to this entry.
    /// # Runtime
    /// O(n) where n is the current number of entries in this partition.
    /// # Returns
    /// If from the Immo a BlockEntry can be constructed then Ok is returned with value being its Index.
    /// If however from the Immo can't be converted to a BlockEntry then Err is returned.
    /// This is the case when the Immo does not plane_position set.
    pub fn add_new_entry_to_block(
        &mut self,
        immo: I,
        block_idx: BlockIdx,
    ) -> BpResult<BlockEntryIdx> {
        let mut entry = BlockEntry::new(immo, block_idx)?;
        let handle = BlockEntryIdx(self.entries.len());

        // compute reachability
        for (other_idx, other) in self
            .entries
            .iter_mut()
            .enumerate()
            .map(|(i, e)| (BlockEntryIdx(i), e))
        {
            if entry.distance(other) <= self.epsilon {
                other.reachable.push(handle);
                entry.reachable.push(other_idx);
            }
        }

        // change cluster
        self.entries.push(entry);
        self[block_idx].entries.push(handle);

        Ok(handle)
    }

    /// Removes entry from its block and adds it to target.
    /// # Runtime
    /// * O(k) where k is the number of entries in the block where entry is initially.
    /// * O(k + k') in debug mode, where k' is the number of entries in the target block.
    /// # Panics
    /// In debug mode this function panics if the entry is already in the target.
    pub fn move_entry_to(&mut self, entry: BlockEntryIdx, target: BlockIdx) {
        debug_assert!(!self[target].entries.contains(&entry));

        let old_block = self[entry].block();
        self[old_block].entries.retain(|&e| e != entry);

        self[target].entries.push(entry);
        self[entry].block = target;
    }

    /// Merge two blocks in the current partition.
    /// # Arguments
    /// * into_idx specifies the BlockIdx, in which all entries from the blocks will end up
    /// * from_idx specifies the BlockIdx, which will be merged into into_idx. It will be empty afterwards.
    pub fn merge_blocks(&mut self, into_idx: BlockIdx, from_idx: BlockIdx) {
        let mut old_entry_idxs = self[from_idx].iter_entries().collect();
        self[into_idx].entries.append(&mut old_entry_idxs);
        self[from_idx].entries = vec![];
    }

    /// remove all Blocks with zero entries
    /// modifies the partition in place
    /// WARNING: invalidates all prior saved indices
    pub fn remove_empty_blocks(&mut self) {
        self.blocks.retain(|block| !block.is_empty());
        let mut entry_new_block_map = HashMap::new();
        self.blocks
            .iter_enumerated()
            .for_each(|(block_idx, block)| {
                block.iter_entries().for_each(|entry_idx| {
                    entry_new_block_map.insert(entry_idx, block_idx);
                })
            });
        entry_new_block_map
            .iter()
            .for_each(|(entry_idx, block_idx)| self[*entry_idx].block = *block_idx);
    }

    /// Creates a JSON string formatted like described in [from_json_file](from_json_file) and writes it to `path`.
    /// # Returns
    /// Returns the string containing the JSON.
    fn create_json_string(&self, collection: Option<String>, query: Option<Document>) -> String {
        let objectid_vec: Vec<Vec<String>> = self
            .iter_blocks()
            .map(|block_idx| {
                self[block_idx]
                    .iter_entries()
                    .map(|entry| self[entry].immo().id().to_string())
                    .collect()
            })
            .collect();

        let json = JsonSchema {
            partition: objectid_vec,
            epsilon: self.epsilon,
            collection,
            query,
        };

        serde_json::to_string(&json).expect("This serialization can't fail by construction")
    }

    /// Creates a JSON value formatted like described in [from_json_file](Partition::from_json_file) and writes it to `path`.
    /// If the file already exists it gets overwritten.
    /// # Returns
    /// Returns an error if writing to the file fails.
    /// Otherwise re&[&'i mut Immo]turns Ok.
    pub fn create_json_file<P: AsRef<Path>>(&self, path: P) -> BpResult<()> {
        self.create_json_file_with_metadata(path, None, None)
    }

    /// Creates a JSON value formatted like described in [from_json_file](Partition::from_json_file) and writes it to `path`.
    /// The JSON will hold additional metadata passed in
    /// If the file already exists it gets overwritten.
    /// # Returns
    /// Returns an error if writing to the file fails.
    /// Otherwise returns Ok.
    pub fn create_json_file_with_metadata<P: AsRef<Path>>(
        &self,
        path: P,
        collection: Option<String>,
        query: Option<Document>,
    ) -> BpResult<()> {
        let json_string = self.create_json_string(collection, query);

        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)
                .expect("Could not create parent directories for JSON output");
        }

        let mut file_handle = fs::File::create(path)?;
        file_handle.write_all(json_string.as_bytes())?;

        Ok(())
    }
}

impl<I: Borrow<Immo>, C: CostFunction> Partition<I, C> {
    /// Creates an empty Partition.
    /// In this matching two entries will be directly reachable, if their plane_distance is at most `epsilon`.
    /// Furthermore for cost calculation the given `cost_function` is used.
    pub fn new(epsilon: f64, cost_function: C) -> Self {
        Self {
            blocks: TiVec::new(),
            entries: TiVec::new(),
            epsilon,
            cost_function,
        }
    }

    /// Reads the file from `path`. This file has to contain a JSON document.
    /// See `specs/ClusterJSON.org` for details.
    /// For `epsilon` and `cost_function` refer to [Partition::new](Partition::new).
    /// # Returns
    /// An error if the file could not be read, or does not contain a valid JSON document.
    /// An error is returned as well, if an ObjectId is malformed, or in `immos` there is no entry with this ObjectId.
    pub fn from_json_file<P>(
        path: P,
        immos: &HashMap<&ObjectId, I>,
        cost_function: C,
    ) -> BpResult<Self>
    where
        I: Clone,
        P: AsRef<Path>,
    {
        // read file
        let mut file_handle = fs::File::open(path)?;
        let json: JsonSchema = serde_json::from_reader(&mut file_handle)?;

        let mut result = Self::new(json.epsilon, cost_function);

        for json_block in json.partition.iter() {
            let cur_block = result.create_new_block();

            for object_id_string in json_block.iter() {
                let object_id = ObjectId::with_string(object_id_string)?;
                let immo = immos.get(&object_id).ok_or("immo not found")?.clone();

                result.add_new_entry_to_block(immo, cur_block)?;
            }
        }
        Ok(result)
    }

    /// Creates a partition from given [Immo]s
    /// Reachability will be calculated
    /// Result will be a partition where every immo is in its own block
    pub fn with_immos(epsilon: f64, cost_function: C, immos: Vec<I>) -> BpResult<Self> {
        let mut part = Self::new(epsilon, cost_function);
        let mut immo_to_block_entry_idx = HashMap::new();
        for immo in immos.into_iter() {
            let immo_id = immo.borrow().id().clone();

            let block_idx = part.create_new_block();
            let entry = BlockEntry::new(immo, block_idx)?;
            let block_entry_idx = part.entries.push_and_get_key(entry);

            part[block_idx].entries.push(block_entry_idx);
            immo_to_block_entry_idx.insert(immo_id, block_entry_idx);
        }

        let mut reachable: HashMap<BlockEntryIdx, Vec<BlockEntryIdx>> = HashMap::new();

        // epsilon + 2.0 is needed to fix rounding errors to guarantee finding too many neighbors.
        // neighbors that are not contained in the radius will be thrown out afterwards.
        sweepline::for_every_close_point_do(
            &part.iter_immos().collect::<Vec<_>>(),
            (epsilon.round() + 2.0) as u64,
            |immo_id, neighbors| {
                for neigh in neighbors.iter() {
                    if *neigh == immo_id {
                        continue;
                    };
                    let immo_entry = &part[*immo_to_block_entry_idx.get(&immo_id).unwrap()];
                    let neigh_entry = &part[*immo_to_block_entry_idx.get(neigh).unwrap()];
                    if immo_entry.distance(neigh_entry) <= epsilon {
                        let other_entry = *immo_to_block_entry_idx.get(&immo_id).unwrap();
                        reachable
                            .entry(other_entry)
                            .or_insert_with(Vec::new)
                            .push(*immo_to_block_entry_idx.get(neigh).unwrap());
                    }
                }
            },
        );

        for (entry_idx, entry_reachable) in reachable.into_iter() {
            part[entry_idx].reachable = entry_reachable;
        }

        Ok(part)
    }
}

impl<I: Borrow<Immo> + Sync, C: CostFunction + Sync> Partition<I, C> {
    /// This function returns the difference in cost of `block_idx`, when `entry_idx` would be added.
    /// This function does not include the cost of removing `entry_idx` from its current block
    /// # Runtime
    /// O(k) where k is the number of entries in the referenced block
    /// # Panics
    /// In debug mode this function panics, if the block of `block_idx` contains `entry_idx`.
    pub fn cost_for_adding_entry_to_block(
        &self,
        entry_idx: BlockEntryIdx,
        block_idx: BlockIdx,
    ) -> f64 {
        debug_assert!(!self[block_idx].entries.contains(&entry_idx));

        let entry = &self[entry_idx];
        self[block_idx]
            .entries
            .iter()
            // we need the factor since this summand will appear twice in the final cost
            .map(|i| 2.0 * self.cost_function.cost(entry.immo(), self[*i].immo()))
            .sum()
    }

    /// returns the cost difference for merging two Blocks
    pub fn cost_for_merging_blocks(&self, block1_idx: BlockIdx, block2_idx: BlockIdx) -> f64 {
        debug_assert!(block1_idx != block2_idx);
        self[block1_idx]
            .iter_par_entries()
            .map(|i| self.cost_for_adding_entry_to_block(i, block2_idx))
            .sum()
    }

    /// This function returns the difference in cost for the current block of `entry_idx`, if we would remove `entry_idx`.
    /// # Runtime
    /// O(k) where k is the number of entries in the current block of `entry_idx`.
    pub fn cost_for_removing_entry_from_current_block(&self, entry_idx: BlockEntryIdx) -> f64 {
        let entry = &self[entry_idx];
        self[entry.block()]
            .entries
            .iter()
            // we need the factore since this summand will appear twice in the final cost
            .map(|i| -2.0 * self.cost_function.cost(entry.immo(), self[*i].immo()))
            .sum()
    }

    /// This function calculates the cost of a given block `block_idx`, calculating all weights of pairs of entries
    /// # Runtime
    /// O(k^2) where k is the number of entries in the block of `block_idx`
    pub fn cost_for_block(&self, block_idx: BlockIdx) -> f64 {
        self[block_idx]
            .iter_entries()
            .map(|entry_idx| -> f64 {
                let entry = &self[entry_idx];
                self[block_idx]
                    .iter_entries()
                    // we need the factor since this summand will appear twice in the final cost
                    .map(|i| {
                        if entry_idx != i {
                            self.cost_function.cost(entry.immo(), self[i].immo())
                        } else {
                            0.0
                        }
                    })
                    .sum()
            })
            .sum()
    }
}

// auxiliary trait implementations
impl<I: Borrow<Immo> + Debug, C: CostFunction> Debug for Partition<I, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Partition")
            .field("blocks", &self.blocks)
            .field("entries", &self.entries)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}

impl<I: Borrow<Immo>, C> Index<BlockIdx> for Partition<I, C> {
    type Output = Block;
    fn index(&self, index: BlockIdx) -> &Self::Output {
        &self.blocks[index]
    }
}

impl<I: Borrow<Immo>, C> IndexMut<BlockIdx> for Partition<I, C> {
    fn index_mut(&mut self, index: BlockIdx) -> &mut Self::Output {
        &mut self.blocks[index]
    }
}

impl<I: Borrow<Immo>, C> Index<BlockEntryIdx> for Partition<I, C> {
    type Output = BlockEntry<I>;
    fn index(&self, index: BlockEntryIdx) -> &Self::Output {
        &self.entries[index]
    }
}

impl<I: Borrow<Immo>, C> IndexMut<BlockEntryIdx> for Partition<I, C> {
    fn index_mut(&mut self, index: BlockEntryIdx) -> &mut Self::Output {
        &mut self.entries[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::immo::ImmoBuilder;
    use cost_functions::ConstantCostFunction;
    use mongodb::bson::doc;
    use proptest::prelude::*;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::PathBuf;
    use test_helpers::*;

    const EXAMPLE_JSON_PATH: &str = "../../data/test_clustering.json";
    const EXAMPLE_JSON_PATH_NO_METADATA: &str = "../../data/test_clustering_no_metadata.json";
    const OUTPUT_JSON_PATH: &str = "../../data/test_clustering_output.json";
    const OUTPUT_JSON_PATH_NO_METADATA: &str = "../../data/test_clustering_output_no_metadata.json";

    fn example_immos() -> Vec<Immo> {
        vec![
            ImmoBuilder::default()
                .id_from_string("5dde57b4c6c36b3bb4fde121")
                .marktwert(5.0)
                .wohnflaeche(100.0)
                .plane_location((1.0, 1.0))
                .build()
                .unwrap(),
            ImmoBuilder::default()
                .id_from_string("5edd91d742936b07d4652e4d")
                .marktwert(5.0)
                .wohnflaeche(100.0)
                .plane_location((1.0, 1.0))
                .build()
                .unwrap(),
            ImmoBuilder::default()
                .id_from_string("5dde4dafc6c36b3bb4f657af")
                .marktwert(5.0)
                .wohnflaeche(100.0)
                .plane_location((1.0, 1.0))
                .build()
                .unwrap(),
        ]
    }

    #[test]
    fn partition_to_json_no_metadata() -> BpResult<()> {
        let output_path = &PathBuf::from(OUTPUT_JSON_PATH_NO_METADATA);

        let immos = example_immos();
        let mut partition = Partition::new(300.0, ConstantCostFunction::discrete());

        let block_zero = partition.create_new_block();
        partition.add_new_entry_to_block(&immos[0], block_zero)?;
        partition.add_new_entry_to_block(&immos[1], block_zero)?;
        let block_one = partition.create_new_block();
        partition.add_new_entry_to_block(&immos[2], block_one)?;

        partition.create_json_file(output_path)?;

        let mut actual = fs::read_to_string(output_path)?;
        actual.push('\n'); // The example is stored with newline at end
        let expected = fs::read_to_string(&PathBuf::from(EXAMPLE_JSON_PATH_NO_METADATA))?;

        assert_eq!(actual, expected);

        fs::remove_file(output_path)?;

        Ok(())
    }

    #[test]
    fn partition_to_json_metadata() -> BpResult<()> {
        let output_path = &PathBuf::from(OUTPUT_JSON_PATH);

        let immos = example_immos();
        let mut partition = Partition::new(300.0, ConstantCostFunction::discrete());

        let block_zero = partition.create_new_block();
        partition.add_new_entry_to_block(&immos[0], block_zero)?;
        partition.add_new_entry_to_block(&immos[1], block_zero)?;
        let block_one = partition.create_new_block();
        partition.add_new_entry_to_block(&immos[2], block_one)?;

        partition.create_json_file_with_metadata(
            output_path,
            Some("ZIMDB_joined".to_string()),
            Some(doc! {
                "kreis": "Berlin, Stadt"
            }),
        )?;

        let mut actual = fs::read_to_string(output_path)?;
        actual.push('\n'); // The example is stored with newline at end
        let expected = fs::read_to_string(&PathBuf::from(EXAMPLE_JSON_PATH))?;

        assert_eq!(actual, expected);

        fs::remove_file(output_path)?;

        Ok(())
    }

    #[test]
    fn partition_from_json() {
        partition_from_json_with_path(&PathBuf::from(EXAMPLE_JSON_PATH));
        partition_from_json_with_path(&PathBuf::from(EXAMPLE_JSON_PATH_NO_METADATA));
    }

    fn partition_from_json_with_path<P: AsRef<Path>>(path: P) {
        let mut id_immo_map = HashMap::new();
        let immos = example_immos();
        for immo in &immos {
            id_immo_map.insert(immo.id(), immo);
        }
        let part = Partition::from_json_file(path, &id_immo_map, ConstantCostFunction::discrete())
            .expect("could not create partition");
        assert_eq!(part.blocks.len(), 2);
        assert_eq!(part.blocks[BlockIdx(0)].entries.len(), 2);
        assert_eq!(part.blocks[BlockIdx(1)].entries.len(), 1);
        assert!((part.epsilon - 300f64).abs() / 300f64 < f64::EPSILON);
    }

    #[test]
    fn partition_add_new_immo_to_block_check_block_size() {
        let mut part = Partition::new(0f64, ConstantCostFunction::discrete());
        let immo = create_new_immo();

        let block = part.create_new_block();
        let block_2 = part.create_new_block();
        assert_eq!(0, part[block].entries.len());
        assert_eq!(0, part[block_2].entries.len());

        part.add_new_entry_to_block(&immo, block).unwrap();
        assert_eq!(1, part[block].entries.len());
        assert_eq!(0, part[block_2].entries.len());

        part.add_new_entry_to_block(&immo, block).unwrap();
        assert_eq!(2, part[block].entries.len());
        assert_eq!(0, part[block_2].entries.len());

        part.add_new_entry_to_block(&immo, block_2).unwrap();
        assert_eq!(2, part[block].entries.len());
        assert_eq!(1, part[block_2].entries.len());
    }

    #[test]
    fn partition_add_new_immo_to_block_check_reachability() {
        let mut part = Partition::new(1.1, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[1.0, 0.0]);
        let immo_c = create_new_immo_at(&[1.0, 1.0]);

        let block = part.create_new_block();
        let a = part.add_new_entry_to_block(&immo_a, block).unwrap();
        let b = part.add_new_entry_to_block(&immo_b, block).unwrap();
        let c = part.add_new_entry_to_block(&immo_c, block).unwrap();

        assert_eq!(part[a].reachable.as_slice(), &[b]);
        assert_eq!(part[c].reachable.as_slice(), &[b]);
        assert!(part[b].reachable.as_slice() == [a, c] || part[b].reachable.as_slice() == [c, a]);
    }

    #[test]
    fn partition_cost_for_adding_entry_to_block() {
        let mut part = Partition::new(0f64, ConstantCostFunction::discrete());

        let immos = [create_new_immo(), create_new_immo(), create_new_immo()];
        let block = part.create_new_block();
        let immo_idx = part.add_new_entry_to_block(&immos[0], block).unwrap();

        let block_2 = part.create_new_block();
        part.add_new_entry_to_block(&immos[1], block_2).unwrap();
        assert!((part.cost_for_adding_entry_to_block(immo_idx, block_2) - 2.0f64).abs() < 1e-9);

        part.add_new_entry_to_block(&immos[2], block_2).unwrap();
        assert!((part.cost_for_adding_entry_to_block(immo_idx, block_2) - 4.0f64).abs() < 1e-9);
    }

    #[test]
    fn partition_cost_removing_entry_from_current_block() {
        let mut part = Partition::new(0f64, ConstantCostFunction::discrete());

        let immos = [create_new_immo(), create_new_immo(), create_new_immo()];
        let block = part.create_new_block();
        let immo_idx = part.add_new_entry_to_block(&immos[0], block).unwrap();
        assert!((part.cost_for_removing_entry_from_current_block(immo_idx) - 0.0).abs() < 1e-9);

        part.add_new_entry_to_block(&immos[1], block).unwrap();
        assert!((part.cost_for_removing_entry_from_current_block(immo_idx) + 2.0).abs() < 1e-9);

        part.add_new_entry_to_block(&immos[2], block).unwrap();
        assert!((part.cost_for_removing_entry_from_current_block(immo_idx) + 4.0).abs() < 1e-9);
    }

    #[test]
    fn create_partition_with_immos() {
        let immo1 = create_new_immo_at(&[0.0, 0.0]);
        let immo2 = create_new_immo_at(&[1.0, 0.0]);
        let immo3 = create_new_immo_at(&[5.0, 0.0]);
        let immos = vec![&immo1, &immo2, &immo3];
        let part = Partition::with_immos(3.0, ConstantCostFunction::discrete(), immos).unwrap();
        assert_eq!(part.blocks.len(), 3);
        for block_idx in part.iter_blocks() {
            assert_eq!(part[block_idx].iter_entries().count(), 1);
            for entry_idx in part[block_idx].iter_entries() {
                if *part[entry_idx].immo().id() == *immo1.id() {
                    assert!(part[entry_idx]
                        .iter_reachable()
                        .all(|entry_idx| part[entry_idx].immo.id() == immo2.id()));
                } else if *part[entry_idx].immo().id() == *immo2.id() {
                    assert!(part[entry_idx]
                        .iter_reachable()
                        .all(|entry_idx| part[entry_idx].immo.id() == immo1.id()));
                } else if *part[entry_idx].immo().id() == *immo3.id() {
                    assert_eq!(part[entry_idx].iter_reachable().count(), 0);
                } else {
                    unreachable!()
                }
            }
        }
    }

    proptest! {
        #[test]
        fn partition_from_immos_reachability_equal_to_manual(immos in prop::collection::vec(full_immo(), 1..128), epsilon in 0.0..2e6) {
            let immo_refs: Vec<_> = immos.iter().collect();
            let part_with_immos = Partition::with_immos(epsilon, ConstantCostFunction::discrete(), immo_refs).unwrap();
            let mut part_manual = Partition::new(epsilon, ConstantCostFunction::discrete());

            let mut entry_to_immo_id_manual = HashMap::new();
            let mut immo_id_to_entry_with_immos = HashMap::new();

            for immo in &immos {
                let block_idx = part_manual.create_new_block();
                let entry_idx = part_manual.add_new_entry_to_block(immo, block_idx).unwrap();
                entry_to_immo_id_manual.insert(entry_idx, immo.id());
            }

            for block_idx in part_with_immos.iter_blocks() {
                for entry_idx in part_with_immos[block_idx].iter_entries() {
                    immo_id_to_entry_with_immos.insert(part_with_immos[entry_idx].immo().id(), entry_idx);
                }
            }

            for block_idx in part_manual.iter_blocks() {
                for entry_idx in part_manual[block_idx].iter_entries() {
                    let mut reachable_manual = HashSet::new();
                    for reachable_idx in part_manual[entry_idx].iter_reachable() {
                        reachable_manual.insert(part_manual[reachable_idx].immo().id());
                    }

                    let mut reachable_with_immos = HashSet::new();
                    for reachable_idx in part_with_immos[immo_id_to_entry_with_immos[part_manual[entry_idx].immo().id()]].iter_reachable() {
                        reachable_with_immos.insert(part_with_immos[reachable_idx].immo().id());
                    }
                    prop_assert_eq!(reachable_with_immos, reachable_manual);
                }
            }

        }
    }

    #[test]
    fn partition_cost_for_block() {
        let mut part = Partition::new(0f64, ConstantCostFunction::discrete());

        let immos = [create_new_immo(), create_new_immo(), create_new_immo()];
        let block = part.create_new_block();
        part.add_new_entry_to_block(&immos[0], block).unwrap();
        assert!((part.cost_for_block(block) - 0.0f64).abs() < 1e-9);

        part.add_new_entry_to_block(&immos[1], block).unwrap();
        assert!((part.cost_for_block(block) - 2.0f64).abs() < 1e-9);

        part.add_new_entry_to_block(&immos[2], block).unwrap();
        assert!((part.cost_for_block(block) - 6.0f64).abs() < 1e-9);
    }

    #[test]
    fn partition_remove_empty_blocks() {
        let mut part = Partition::new(0f64, ConstantCostFunction::discrete());

        let immos = [create_new_immo(), create_new_immo(), create_new_immo()];
        let blocks = [
            part.create_new_block(),
            part.create_new_block(),
            part.create_new_block(),
            part.create_new_block(),
        ];
        part.add_new_entry_to_block(&immos[0], blocks[0]).unwrap();
        part.add_new_entry_to_block(&immos[1], blocks[2]).unwrap();
        part.add_new_entry_to_block(&immos[2], blocks[2]).unwrap();

        part.remove_empty_blocks();
        assert_eq!(part.blocks.len(), 2);
        for block_idx in part.iter_blocks() {
            for entry in part[block_idx].iter_entries() {
                assert_eq!(part[entry].block(), block_idx);
            }
        }
    }
}
