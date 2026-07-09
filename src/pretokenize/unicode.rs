use icu::properties::props::{EnumeratedProperty, GeneralCategory, GeneralCategoryGroup, WhiteSpace};
use icu::properties::CodePointSetData;

#[inline]
pub(crate) fn get_general_category(c: char) -> GeneralCategory {
    GeneralCategory::for_char(c)
}

#[inline]
pub(crate) fn is_gc_letter(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Letter.contains(gc)
}

#[inline]
pub(crate) fn is_gc_number(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Number.contains(gc)
}

/// Unicode White_Space property — matches the same characters as `\s` in regex.
/// This includes GeneralCategory::Separator (Zs/Zl/Zp) PLUS control characters
/// like U+0009 (TAB), U+000A (LF), U+000D (CR), U+0085 (NEL), etc.
#[inline]
pub(crate) fn is_whitespace(c: char) -> bool {
    // The set is a static compiled-data lookup, but cache the borrowed handle
    // to avoid repeated constructor overhead.
    static WS: std::sync::LazyLock<icu::properties::CodePointSetDataBorrowed<'static>> =
        std::sync::LazyLock::new(CodePointSetData::new::<WhiteSpace>);
    WS.contains(c)
}

#[inline]
pub(crate) fn is_letter(c: char) -> bool {
    is_gc_letter(get_general_category(c))
}

#[inline]
pub(crate) fn is_number(c: char) -> bool {
    is_gc_number(get_general_category(c))
}

#[inline]
pub(crate) fn is_letter_complete(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    is_letter(c)
}

#[inline]
pub(crate) fn is_number_complete(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    is_number(c)
}

/// Whitespace check using the Unicode White_Space property.
/// For ASCII, uses the fast `is_ascii_whitespace()` path.
/// For non-ASCII, checks the ICU White_Space property which includes
/// U+0085 (NEL), U+00A0 (NBSP), U+2000-U+200A, etc.
#[inline]
pub(crate) fn is_separator_complete(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_whitespace();
    }
    is_whitespace(c)
}

#[inline]
pub(crate) fn is_other_complete(c: char) -> bool {
    if c.is_ascii() {
        return !c.is_ascii_alphanumeric() && !c.is_ascii_whitespace();
    }
    let gc = get_general_category(c);
    !is_gc_letter(gc) && !is_gc_number(gc) && !is_whitespace(c)
}

// ---------------------------------------------------------------------------
// Packed codepoint → class table (hot-path classification)
// ---------------------------------------------------------------------------

/// Character class as used by the pretokenization regexes: `\p{L}`, `\p{N}`,
/// `\s` (White_Space), and everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Other = 3,
}

/// 2-bit class per codepoint, 4 codepoints per byte (~272 KiB total).
/// A single L1 load replaces the ICU GeneralCategory trie walk plus the
/// White_Space set binary search that the `is_*` predicates above pay per
/// call. Only the cache lines for scripts actually present in the input
/// stay resident.
static CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_class_table);

fn build_class_table() -> Box<[u8]> {
    use icu::properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![CharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (group, class) in [
        (GeneralCategoryGroup::Letter, CharClass::Letter),
        (GeneralCategoryGroup::Number, CharClass::Number),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from GC Letter/Number, so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize].fill(CharClass::Whitespace as u8);
    }
    classes
        .as_chunks::<4>().0.iter()
        .map(|c| c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6))
        .collect()
}

/// Classify a codepoint with one table load. `cp` must be a valid scalar
/// value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn class_of(cp: u32) -> CharClass {
    debug_assert!(cp < 0x110000);
    let byte = unsafe { *CLASS_TABLE.get_unchecked((cp >> 2) as usize) };
    match (byte >> ((cp & 3) << 1)) & 3 {
        0 => CharClass::Letter,
        1 => CharClass::Number,
        2 => CharClass::Whitespace,
        _ => CharClass::Other,
    }
}

// ---------------------------------------------------------------------------
// DeepSeek character classes (finer split of `Other`)
// ---------------------------------------------------------------------------

/// Character class as used by the DeepSeek V3 main regex, which additionally
/// distinguishes `\p{M}` (joins letter runs) and `\p{P}`/`\p{S}` (punctuation
/// runs) from the remaining `Other` codepoints (controls, format chars,
/// unassigned), which the regex leaves unmatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum DsCharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Mark = 3,
    PunctSym = 4,
    Other = 5,
}

/// Four-class view for schemes whose regex joins `\p{M}` into letter
/// runs and excludes it from punctuation runs (Qwen3.5's
/// `[\p{L}\p{M}]+` / `[^\s\p{L}\p{M}\p{N}]+`): marks classify as
/// letters, everything else as in [`class_of`].
#[inline(always)]
pub(crate) fn class_of_marks_join(cp: u32) -> CharClass {
    match ds_class_of(cp) {
        DsCharClass::Letter | DsCharClass::Mark => CharClass::Letter,
        DsCharClass::Number => CharClass::Number,
        DsCharClass::Whitespace => CharClass::Whitespace,
        DsCharClass::PunctSym | DsCharClass::Other => CharClass::Other,
    }
}

/// 4-bit class per codepoint, 2 codepoints per byte (~544 KiB total).
static DS_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_ds_class_table);

fn build_ds_class_table() -> Box<[u8]> {
    use icu::properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![DsCharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (group, class) in [
        (GeneralCategoryGroup::Letter, DsCharClass::Letter),
        (GeneralCategoryGroup::Number, DsCharClass::Number),
        (GeneralCategoryGroup::Mark, DsCharClass::Mark),
        (GeneralCategoryGroup::Punctuation, DsCharClass::PunctSym),
        (GeneralCategoryGroup::Symbol, DsCharClass::PunctSym),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from the groups above except Zs/Zl/Zp (which
    // are in none of them), so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(DsCharClass::Whitespace as u8);
    }
    classes
        .as_chunks::<2>().0.iter()
        .map(|c| c[0] | (c[1] << 4))
        .collect()
}

/// Classify a codepoint for the DeepSeek scheme with one table load. `cp`
/// must be a valid scalar value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn ds_class_of(cp: u32) -> DsCharClass {
    debug_assert!(cp < 0x110000);
    let byte = unsafe { *DS_CLASS_TABLE.get_unchecked((cp >> 1) as usize) };
    match (byte >> ((cp & 1) << 2)) & 0xF {
        0 => DsCharClass::Letter,
        1 => DsCharClass::Number,
        2 => DsCharClass::Whitespace,
        3 => DsCharClass::Mark,
        4 => DsCharClass::PunctSym,
        _ => DsCharClass::Other,
    }
}

/// The CJK ranges isolated by the DeepSeek pretokenizer's second Split:
/// `[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]` (CJK unified
/// ideographs, hiragana, katakana — the two kana blocks are contiguous).
#[inline(always)]
pub(crate) fn is_deepseek_cjk(cp: u32) -> bool {
    (0x4E00..=0x9FA5).contains(&cp) || (0x3040..=0x30FF).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed table must agree with the ICU predicates for every scalar.
    #[test]
    fn class_table_matches_icu() {
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let expected = if is_letter(c) {
                CharClass::Letter
            } else if is_number(c) {
                CharClass::Number
            } else if is_whitespace(c) {
                CharClass::Whitespace
            } else {
                CharClass::Other
            };
            assert_eq!(class_of(cp), expected, "mismatch at U+{cp:04X}");
        }
    }

    /// The DeepSeek table must agree with ICU for every scalar, and refine
    /// `class_of` (identical on Letter/Number/Whitespace).
    #[test]
    fn ds_class_table_matches_icu() {
        for cp in 0..=char::MAX as u32 {
            let Some(c) = char::from_u32(cp) else { continue };
            let gc = get_general_category(c);
            let expected = if is_gc_letter(gc) {
                DsCharClass::Letter
            } else if is_gc_number(gc) {
                DsCharClass::Number
            } else if is_whitespace(c) {
                DsCharClass::Whitespace
            } else if GeneralCategoryGroup::Mark.contains(gc) {
                DsCharClass::Mark
            } else if GeneralCategoryGroup::Punctuation.contains(gc)
                || GeneralCategoryGroup::Symbol.contains(gc)
            {
                DsCharClass::PunctSym
            } else {
                DsCharClass::Other
            };
            assert_eq!(ds_class_of(cp), expected, "mismatch at U+{cp:04X}");
        }
    }
}
