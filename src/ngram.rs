//! n-gram (prompt-lookup) draft proposals.
//!
//! Speculation is only worth its verification cost when the draft itself is
//! nearly free. This drafter never runs a model: it keeps an incremental index
//! over the tokens already in context and, when the recent token tail repeats
//! an earlier tail, proposes the continuation that followed it last time. On a
//! miss it proposes nothing and the caller decodes normally, so a workload
//! with no repetition pays only the lookup.
//!
//! Correctness never depends on the hash. A lookup verifies the candidate's
//! actual token IDs before proposing anything, so a colliding key produces a
//! miss rather than a wrong draft. Draft tokens are proposals in any case: the
//! target model verifies every one of them.

use std::collections::HashMap;

/// n-gram lengths indexed incrementally. Longer keys are more selective, so a
/// lookup tries them longest-first.
pub const INDEXED_MATCH_LENGTHS: [usize; 3] = [4, 3, 2];

/// Longest and shortest key this index maintains.
pub const MAX_MATCH_LEN: usize = 4;
pub const MIN_MATCH_LEN: usize = 2;

/// Hash of the token IDs forming one n-gram key.
pub type TokenHasher = fn(&[u32]) -> u64;

/// FNV-1a over the raw token IDs. Deterministic across runs and platforms,
/// which keeps drafting reproducible.
pub fn fnv1a_tokens(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &token in tokens {
        hash ^= u64::from(token);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A draft proposal and the match that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramDraft {
    pub tokens: Vec<u32>,
    /// Key length that matched.
    pub match_len: usize,
    /// Position of the last token of the earlier occurrence.
    pub source_position: usize,
    /// Whether a stop token truncated the proposal.
    pub truncated_at_stop: bool,
}

/// Incremental n-gram index over the full token sequence in context.
///
/// One map per indexed key length holds the most recent position at which that
/// key ended. `last_match` caches, per key length, the most recent occurrence
/// *before* the current suffix, which is exactly what a drafter standing at
/// the end of the sequence needs; it is filled during `push`, so a lookup is a
/// slice comparison and nothing more.
#[derive(Debug, Clone)]
pub struct NgramIndex {
    tokens: Vec<u32>,
    maps: [HashMap<u64, usize>; INDEXED_MATCH_LENGTHS.len()],
    last_match: [Option<usize>; INDEXED_MATCH_LENGTHS.len()],
    hasher: TokenHasher,
}

impl Default for NgramIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramIndex {
    pub fn new() -> Self {
        Self::with_hasher(fnv1a_tokens)
    }

    /// Build an index with a custom key hash. Used by tests to force
    /// collisions and prove the verified token comparison rejects them.
    pub fn with_hasher(hasher: TokenHasher) -> Self {
        Self {
            tokens: Vec::new(),
            maps: Default::default(),
            last_match: [None; INDEXED_MATCH_LENGTHS.len()],
            hasher,
        }
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        for map in &mut self.maps {
            map.clear();
        }
        self.last_match = [None; INDEXED_MATCH_LENGTHS.len()];
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn extend(&mut self, tokens: &[u32]) {
        for &token in tokens {
            self.push(token);
        }
    }

    /// Append one committed token. Constant work: one map insert per indexed
    /// key length.
    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
        let end = self.tokens.len() - 1;
        for (slot, &match_len) in INDEXED_MATCH_LENGTHS.iter().enumerate() {
            if self.tokens.len() < match_len {
                self.last_match[slot] = None;
                continue;
            }
            let key = (self.hasher)(&self.tokens[self.tokens.len() - match_len..]);
            // The entry present before this insert is the most recent earlier
            // occurrence of the suffix that now ends the sequence.
            self.last_match[slot] = self.maps[slot].insert(key, end);
        }
    }

    /// Most recent verified occurrence of the current `match_len`-token suffix,
    /// reported as the position of that occurrence's last token.
    ///
    /// Returns `None` when the key is absent, when the stored position lost a
    /// hash collision, or when the sequence is shorter than the key.
    pub fn most_recent_match(&self, match_len: usize) -> Option<usize> {
        let slot = INDEXED_MATCH_LENGTHS
            .iter()
            .position(|&len| len == match_len)?;
        let candidate = self.last_match[slot]?;
        if candidate + 1 < match_len || candidate >= self.tokens.len() {
            return None;
        }
        let suffix = &self.tokens[self.tokens.len() - match_len..];
        let earlier = &self.tokens[candidate + 1 - match_len..=candidate];
        // Correctness must not depend on hash quality: compare the tokens.
        (earlier == suffix).then_some(candidate)
    }

    /// Propose up to `draft_len` continuation tokens for the current suffix.
    ///
    /// Key lengths are tried longest-first down to `min_match`; the first one
    /// with a verified match wins. The proposal is truncated at the first stop
    /// token it contains, keeping the tokens before it — verification decides
    /// whether the stop is reached.
    pub fn draft(
        &self,
        draft_len: usize,
        min_match: usize,
        is_stop: impl Fn(u32) -> bool,
    ) -> Option<NgramDraft> {
        if draft_len == 0 {
            return None;
        }
        let min_match = min_match.clamp(MIN_MATCH_LEN, MAX_MATCH_LEN);
        for &match_len in INDEXED_MATCH_LENGTHS
            .iter()
            .filter(|&&len| len >= min_match)
        {
            let Some(position) = self.most_recent_match(match_len) else {
                continue;
            };
            let start = position + 1;
            let end = (start + draft_len).min(self.tokens.len());
            if start >= end {
                continue;
            }
            let candidate = &self.tokens[start..end];
            let stop = candidate.iter().position(|&token| is_stop(token));
            let tokens = candidate[..stop.unwrap_or(candidate.len())].to_vec();
            if tokens.is_empty() {
                // The continuation begins with a stop token; there is nothing
                // to verify beyond what the target model will decide itself.
                return None;
            }
            return Some(NgramDraft {
                tokens,
                match_len,
                source_position: position,
                truncated_at_stop: stop.is_some(),
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index_of(tokens: &[u32]) -> NgramIndex {
        let mut index = NgramIndex::new();
        index.extend(tokens);
        index
    }

    #[test]
    fn lookup_returns_the_most_recent_prior_occurrence() {
        // "1 2 3" occurs at positions ending 2 and 6; the suffix repeats it.
        let index = index_of(&[1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3]);
        assert_eq!(index.most_recent_match(3), Some(6));
        assert_eq!(index.most_recent_match(2), Some(6));
    }

    #[test]
    fn draft_continues_from_the_most_recent_occurrence() {
        let index = index_of(&[1, 2, 3, 40, 41, 42, 9, 1, 2, 3, 50, 51, 52, 1, 2, 3]);
        let draft = index.draft(3, 3, |_| false).unwrap();
        assert_eq!(draft.tokens, [50, 51, 52]);
        assert_eq!(index.tokens()[draft.source_position], 3);
    }

    #[test]
    fn longer_keys_are_preferred_over_shorter_ones() {
        // Suffix "7 2 3": the "2 3" bigram last occurred at a different place
        // than the "7 2 3" trigram, so the trigram's continuation must win.
        let index = index_of(&[7, 2, 3, 60, 5, 2, 3, 70, 7, 2, 3]);
        let draft = index.draft(1, 3, |_| false).unwrap();
        assert_eq!(draft.match_len, 3);
        assert_eq!(draft.tokens, [60]);
        let shorter = index.draft(1, 2, |_| false).unwrap();
        assert_eq!(shorter.match_len, 3, "longest key still wins when it hits");
    }

    #[test]
    fn min_match_two_finds_matches_a_trigram_key_would_miss() {
        let index = index_of(&[4, 5, 99, 1, 2, 4, 5]);
        assert!(index.draft(2, 3, |_| false).is_none());
        let draft = index.draft(2, 2, |_| false).unwrap();
        assert_eq!(draft.match_len, 2);
        assert_eq!(draft.tokens, [99, 1]);
    }

    #[test]
    fn draft_truncates_at_the_first_stop_token() {
        let index = index_of(&[1, 2, 3, 40, 41, 77, 42, 1, 2, 3]);
        let draft = index.draft(5, 3, |token| token == 77).unwrap();
        assert_eq!(draft.tokens, [40, 41]);
        assert!(draft.truncated_at_stop);
    }

    #[test]
    fn draft_is_empty_when_the_continuation_starts_with_a_stop_token() {
        let index = index_of(&[1, 2, 3, 77, 40, 1, 2, 3]);
        assert!(index.draft(5, 3, |token| token == 77).is_none());
    }

    #[test]
    fn a_sequence_without_repetition_drafts_nothing() {
        let index = index_of(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(index.draft(4, 3, |_| false).is_none());
        assert!(index.draft(4, 2, |_| false).is_none());
    }

    #[test]
    fn hash_collisions_are_rejected_by_comparing_token_ids() {
        // Every key hashes to the same bucket, so the map hands back a
        // position whose tokens do not match the current suffix.
        let mut index = NgramIndex::with_hasher(|_| 0);
        index.extend(&[1, 2, 3, 40, 41, 42, 7, 8, 9]);
        assert_eq!(index.most_recent_match(3), None);
        assert!(index.draft(4, 3, |_| false).is_none());

        // A colliding key costs a match it would otherwise have found: the
        // real occurrence of "1 2 3" at position 2 was overwritten by later
        // trigrams sharing the bucket. Degrading to a miss is the required
        // behavior; proposing position 5's tokens would not be.
        let mut overwritten = NgramIndex::with_hasher(|_| 0);
        overwritten.extend(&[1, 2, 3, 40, 41, 1, 2, 3]);
        assert_eq!(overwritten.most_recent_match(3), None);
        assert!(overwritten.draft(2, 3, |_| false).is_none());

        // Verified equality still accepts a genuine match under the same
        // degenerate hash when the stored position's tokens do compare equal.
        let mut periodic = NgramIndex::with_hasher(|_| 0);
        periodic.extend(&[5, 5, 5, 5]);
        assert_eq!(periodic.most_recent_match(3), Some(2));
        assert_eq!(periodic.draft(1, 3, |_| false).unwrap().tokens, [5]);
    }

    #[test]
    fn draft_is_clamped_to_the_tokens_that_exist() {
        let index = index_of(&[1, 2, 3, 40, 1, 2, 3]);
        let draft = index.draft(8, 3, |_| false).unwrap();
        assert_eq!(draft.tokens, [40, 1, 2, 3]);
    }

    #[test]
    fn clear_drops_every_indexed_position() {
        let mut index = index_of(&[1, 2, 3, 40, 1, 2, 3]);
        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.most_recent_match(3), None);
        index.extend(&[1, 2, 3]);
        assert_eq!(index.most_recent_match(3), None);
    }

    #[test]
    fn zero_draft_length_proposes_nothing() {
        let index = index_of(&[1, 2, 3, 40, 1, 2, 3]);
        assert!(index.draft(0, 3, |_| false).is_none());
    }
}
