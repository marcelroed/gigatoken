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
    CodePointSetData::new::<WhiteSpace>().contains(c)
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
