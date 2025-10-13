use crate::input::DocRef;
use crate::pretokenize::pretoken::Pretoken;
use crate::pretokenize::{pretokenize_as_iter, PretokenizerIter};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash},
    ops::AddAssign,
};

pub(crate) trait ParallelPretokenCountable<'a, S: BuildHasher + Default> {
    /// Count pretokens and
    /// Should only be used with chunked parallel iterators, meaning where the number of parallel elements ≈ number of threads
    fn par_pretoken_count(self) -> HashMap<Pretoken<'a>, usize, S>;
}

pub(crate) trait PretokenCountable<'a, S: BuildHasher + Default> {
    fn pretoken_count(self) -> HashMap<Pretoken<'a>, usize, S>;
}

pub(crate) trait PretokenCountableWeighted<'a, S: BuildHasher + Default> {
    fn pretoken_count(self) -> HashMap<Pretoken<'a>, usize, S>;
}

impl<'a, T, S> PretokenCountable<'a, S> for T
where
    T: Iterator<Item = Pretoken<'a>>,
    S: BuildHasher + Default,
{
    fn pretoken_count(self) -> HashMap<Pretoken<'a>, usize, S> {
        self.fold(HashMap::default(), |mut counts, token| {
            *counts.entry(token).or_default() += 1;
            counts
        })
    }
}

impl<'a, T, I, S> ParallelPretokenCountable<'a, S> for I
where
    I: ParallelIterator<Item = T>,
    T: Iterator<Item = Pretoken<'a>>,
    S: BuildHasher + Default + Send,
{
    fn par_pretoken_count(self) -> HashMap<Pretoken<'a>, usize, S> {
        self.map(PretokenCountable::pretoken_count)
            .par_merge_counts()
    }
}

impl<'a, T, S> PretokenCountableWeighted<'a, S> for T
where
    T: Iterator<Item = (DocRef<'a>, usize)>,
    S: BuildHasher + Default,
{
    fn pretoken_count(self) -> HashMap<Pretoken<'a>, usize, S> {
        let mut hashmap = HashMap::default();
        self.map(|doc| (pretokenize_as_iter(doc.0.as_ref()), doc.1))
            .for_each(|(pretoken_iter, count): (PretokenizerIter, usize)| {
                pretoken_iter.for_each(|pretoken| {
                    hashmap
                        .entry(pretoken)
                        .and_modify(|e| *e += count)
                        .or_insert(count);
                });
            });
        hashmap
    }
}

pub(crate) trait ParallelMergeCounts<K, V, S> {
    fn par_merge_counts(self) -> HashMap<K, V, S>;
}

impl<T, K, V, S> ParallelMergeCounts<K, V, S> for T
where
    T: ParallelIterator<Item = HashMap<K, V, S>>,
    K: Eq + Hash,
    V: AddAssign + Default,
    S: BuildHasher + Default,
{
    fn par_merge_counts(self) -> HashMap<K, V, S> {
        self.reduce(HashMap::default, |mut acc, counts| {
            if acc.is_empty() {
                return counts;
            }

            for (k, v) in counts {
                *acc.entry(k).or_default() += v;
            }
            acc
        })
    }
}
