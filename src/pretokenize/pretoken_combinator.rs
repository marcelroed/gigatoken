//! Implement the regex
//! '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
//! using winnow parser combinators.
use crate::pretokenize::{unicode, Pretoken};
use std::cmp::min;

use eyre::Context;
use itertools::Itertools;
use rayon::prelude::*;
use winnow::combinator::{alt, iterator, opt, peek, repeat_till, trace};
use winnow::prelude::*;
use winnow::token::{one_of, take_while};
use winnow::Parser;

fn contraction(input: &mut &str) -> ModalResult<()> {
    ('\'', alt(("s", "d", "m", "t", "ll", "ve", "re")))
        .void()
        .parse_next(input)
}

// fn letter<'a>(input: &mut &'a str) -> ModalResult<()> {
//     let slice = &input[..];
//     unicode::is_letter.void().parse_next(input)
// }

fn letter_run(input: &mut &str) -> ModalResult<()> {
    trace(
        "letter_run",
        (opt(' '), take_while(1.., unicode::is_letter_complete)),
    )
    .void()
    .parse_next(input)
}

fn number_run(input: &mut &str) -> ModalResult<()> {
    trace(
        "number_run",
        (opt(' '), take_while(1.., unicode::is_number_complete)),
    )
    .void()
    .parse_next(input)
}

fn whitespace_run(input: &mut &str) -> ModalResult<()> {
    trace(
        "whitespace_run",
        repeat_till::<_, (), (), _, _, _, _>(
            1..,
            one_of(unicode::is_separator_complete).void(),
            peek((
                one_of(unicode::is_separator_complete),
                one_of(|c| !unicode::is_separator_complete(c)),
            ))
            .void(),
        ),
    )
    .void()
    .parse_next(input)
}

fn single_whitespace(input: &mut &str) -> ModalResult<()> {
    trace("single_whitespace", one_of(unicode::is_separator_complete))
        .void()
        .parse_next(input)
}

fn other_run(input: &mut &str) -> ModalResult<()> {
    trace(
        "other_run",
        (opt(' '), take_while(1.., |c| unicode::is_other_complete(c))),
    )
    .void()
    .parse_next(input)
}

fn pretoken<'a>(input: &mut &'a str) -> ModalResult<Pretoken<'a>> {
    alt((
        contraction,
        letter_run,
        number_run,
        other_run,
        whitespace_run,
        single_whitespace,
    ))
    .take()
    .map(|s: &str| Pretoken(s.as_bytes()))
    .parse_next(input)
}

// pub fn pretokens<'a>(input: &mut &'a str) -> ModalResult<Vec<&'a str>> {
//     repeat::<_, &str, Vec<&str>, _, _>(1.., pretoken).parse_next(input)
// }

// pub fn parse_pretokens(input: &[u8]) -> Result<Vec<&str>> {
//     let mut slice: &str = unsafe { std::str::from_utf8_unchecked(input) };
//     let result = pretokens(&mut slice).map_err(|e| anyhow!("Parse error: {}", e));
//     if slice.len() != 0 {
//         Err(anyhow!(
//             "Did not consume all input, remaining: {:?}",
//             &slice[..min(32, slice.len())]
//         ))
//     } else {
//         result
//     }
// }

pub struct PretokenIterator<'a> {
    input: &'a [u8],
}

pub fn pretokens_iterator<'a>(
    input: &'a str,
) -> winnow::combinator::ParserIterator<
    impl FnMut(
        &mut &'a str,
    )
        -> std::result::Result<Pretoken<'a>, winnow::error::ErrMode<winnow::error::ContextError>>
    + 'a,
    &'a str,
    Pretoken<'a>,
    winnow::error::ErrMode<winnow::error::ContextError>,
> {
    iterator(input, pretoken)
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_parse_pretokens() {
    //     let input = "Hello, world!";
    //     let pretokens = parse_pretokens(input.as_bytes()).unwrap();
    //     eprintln!("{:?}", pretokens);
    // }

    #[test]
    fn combinator_compare() {
        let input =
            std::fs::read_to_string("/Users/marcel/data/TinyStoriesV2-GPT4-valid.txt").unwrap();
        let input_bytes = input.as_bytes();
        let standard_iterator = crate::pretokenize::pretokenize_as_iter(input_bytes);
        let mut combinator_iterator = pretokens_iterator(&input);
        for eorb in standard_iterator.zip_longest(&mut combinator_iterator) {
            match eorb {
                itertools::EitherOrBoth::Both(a, b) => {
                    if a.0 != b.0 {
                        eprintln!(
                            "Mismatch: {:?} != {:?}",
                            String::from_utf8_lossy(a.0),
                            String::from_utf8_lossy(b.0)
                        );

                        // Find text before and after the mismatch by comparing pointers from a.0 and input_bytes
                        let a_start = a.0.as_ptr() as usize;
                        let b_start = b.0.as_ptr() as usize;
                        let input_start = input_bytes.as_ptr() as usize;
                        let a_offset = a_start - input_start;

                        let region = &input_bytes
                            [a_offset.saturating_sub(32)..min(input_bytes.len(), a_offset + 32)];
                        eprintln!("Context: {:?}", String::from_utf8_lossy(region));

                        assert!(false);
                    }
                }
                itertools::EitherOrBoth::Left(a) => {
                    eprintln!("Left only: {:?}", String::from_utf8_lossy(a.0));

                    // Find text before and after the mismatch by comparing pointers from a.0 and input_bytes
                    let a_start = a.0.as_ptr() as usize;
                    let input_start = input_bytes.as_ptr() as usize;
                    let a_offset = a_start - input_start;

                    let region = &input_bytes
                        [a_offset.saturating_sub(32)..min(input_bytes.len(), a_offset + 32)];
                    eprintln!("Context: {:?}", String::from_utf8_lossy(region));

                    assert!(false);
                }
                itertools::EitherOrBoth::Right(b) => {
                    eprintln!("Right only: {:?}", String::from_utf8_lossy(b.0));

                    // Find text before and after the mismatch by comparing pointers from b and input_bytes
                    let b_start = b.as_ptr() as usize;
                    let input_start = input_bytes.as_ptr() as usize;
                    let b_offset = b_start - input_start;

                    let region = &input_bytes
                        [b_offset.saturating_sub(32)..min(input_bytes.len(), b_offset + 32)];
                    eprintln!("Context: {:?}", String::from_utf8_lossy(region));

                    assert!(false);
                }
            }
        }
    }
}
