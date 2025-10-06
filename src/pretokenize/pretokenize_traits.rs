use std::collections::HashMap;

use crate::pretokenize::pretokenize_as_iter;

trait Pretokenizeable<'a> {
    fn pretokenize(self) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher>;
}

trait PretokenizeableWeighted<'a> {
    fn pretokenize(self) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher>;
}

impl<'a, T> Pretokenizeable<'a> for T
where
    T: Iterator<Item = &'a [u8]>,
{
    fn pretokenize(self) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher> {
        let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        self.flat_map(|doc| pretokenize_as_iter(doc))
            .for_each(|token| {
                hashmap.entry(token).and_modify(|e| *e += 1).or_insert(1);
            });
        hashmap
    }
}

impl<'a, T> PretokenizeableWeighted<'a> for T
where
    T: Iterator<Item = (&'a [u8], usize)>,
{
    fn pretokenize(self) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher> {
        let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        self.map(|doc| (pretokenize_as_iter(doc.0), doc.1))
            .for_each(|(pretoken_iter, count)| {
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
