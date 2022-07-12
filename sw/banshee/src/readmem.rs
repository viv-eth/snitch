use std::{
    str::FromStr,
    collections::HashMap,
    fs::File,
    fs,
    io::Write,
    io::prelude::*,
    num::ParseIntError,
    num::ParseFloatError,
};

use pest::{iterators::Pair, Parser};
use pest_derive::Parser;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("syntax error")]
    ParseError(#[from] pest::error::Error<Rule>),
    #[error("we didn't successfully parse a top level syntax object")]
    ParseFailure,
    #[error("numeric format error")]
    NumericFormatError(#[from] std::num::ParseIntError),
    #[error("float format error")]
    FloatFormatError(#[from] std::num::ParseFloatError),
}

/// The type of content contained in a file to be read.
pub enum ContentType {
    /// The file is compatible with $readmemh and contains hex values
    Hex,
    /// The file is compatible with $readmemb and contains binary values
    Binary,
    /// Add floating point because floating point to hex conversion just sucks
    Float,
}


#[derive(Parser)]
#[grammar = "grammar.pest"]
struct ReadmemParser;

#[derive(Debug)]
enum Item<I> {
    // addressing in banshee in u64 format
    Address(u64),
    Number(I),
}

/// A trait implemented on unsigned numeric types to allow us to be polymorphic
/// over them, supporting any storage type.
#[doc(hidden)]
pub trait Integral: Sized {
    fn from_str_radix(src: &str, radix: u32) -> Result<Self, std::num::ParseIntError>;
    fn from_str_float(src: &str) -> Result<Self, std::num::ParseFloatError>;
    fn zero() -> Self;
}

macro_rules! integrate {
    ($t:ty) => {
        impl Integral for $t {
            fn from_str_radix(src: &str, radix: u32) -> Result<Self, std::num::ParseIntError> {
                <$t>::from_str_radix(src, radix)
            }
            fn zero() -> Self {
                0
            }
            fn from_str_float(src: &str) -> Result<Self, std::num::ParseFloatError> {
                let str_float: f32 = src.parse().unwrap();
                // trace!("Before transmute: {}", str_float);
                let tx_float = unsafe { std::mem::transmute::<f32, u32>(str_float) };
                // trace!("After transmute: {}", tx_float);
                let float = unsafe { std::mem::transmute::<u32, f32>(tx_float) };
                // trace!("After after transmute retransmute: {}", float);
                Ok(tx_float)
            }
        }
    };
}

//integrate!(u8);
//integrate!(u16);
integrate!(u32); // --> For Occamy we only save the values in u32 in DRAM
//integrate!(u64);
//integrate!(u128);

/// Parse a `Pair` into an AST `Item`.
fn parse_value<I>(pair: Pair<Rule>) -> Result<Item<I>, Error>
where
    I: Integral,
{
    let is_zx = |c| c == 'x' || c == 'z' || c == 'X' || c == 'Z';
    Ok(match pair.as_rule() {
        // checks for address
        Rule::addr => Item::Address(u64::from_str_radix(&pair.as_str()[1..], 16)?),
        // checks for value
        Rule::hex => {
            let without_underscore = pair.as_str().replace("_", "");
            let without_zx = without_underscore.replace(is_zx, "0");
            Item::Number(Integral::from_str_radix(&without_zx, 16)?)
        }
        Rule::bin => {
            let without_underscore = pair.as_str().replace("_", "");
            let without_zx = without_underscore.replace(is_zx, "0");
            Item::Number(Integral::from_str_radix(&without_zx, 2)?)
        }
        Rule::float => {
            // trace!("Parsing float: {}", pair.as_str());
            let without_underscore = pair.as_str().replace("_", "");
            let without_zx = without_underscore.replace(is_zx, "0");
            Item::Number(Integral::from_str_float(&without_zx)?)
        }
        r => unreachable!(
            "should not hit this rule {:?}, all our other rules are silent",
            r
        ),
    })
}

// we need to decode the hex values to properly handle the memory writing 
// TODO: implement this for general cases
// pub fn decode_hex(s: &str) -> Result<Vec<u8>, ParseIntError> {
//     (0..s.len())
//         .step_by(2)
//         .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
//         .collect()
// }

// pub fn encode_hex(bytes: &[u8]) -> String {
//     let mut s = String::with_capacity(bytes.len() * 2);
//     for &b in bytes {
//         write!(&mut s, "{:02x}", b).unwrap();
//     }
//     s
// }

pub fn readmem<I>(content: &str, content_type: ContentType) -> Result<HashMap<u64, I>, Error>
where
    I: Integral + FromStr + Clone + std::fmt::LowerHex + Copy + std::fmt::Display, //INFO: VIVI EDIT
    // I: Integral,
{

    let rule = match content_type {
        ContentType::Hex => Rule::readmemh,
        ContentType::Binary => Rule::readmemb,
        ContentType::Float => Rule::readmemf,
    };
    // trace!("Start parsing with rule: {:?}", rule);
    // trace!("Parsing content: {}", content);
    let content = ReadmemParser::parse(rule, content)?;
    let mut result = HashMap::<u64, I>::new();
    // initial position is at zero
    // if Address of Item is non-zero it will be overwritten
    let mut pos = 0;
    for val in content {
        if let Rule::EOI = val.as_rule() {
            continue;
        }
        // we check whether the values in the 
        // file we provide actually matches with 
        // the rules defined in the parser 
        let val = parse_value::<I>(val)?;
        match val {
            // 
            Item::Address(a) => {
                pos = a;
            },
            Item::Number(n) => {
                // if pos + 1 >= result.len() {
                //     result.resize(pos + 1, I::zero());
                // } --> this fills up the values up to zero for a VECTOR
                //
                //result[pos] = n;
                // test.insert(
                //     pos, n
                // );
                // for (key, value) in test.iter(){
                //     println!("addr = 0x{:x}, value = {:x}", key, value);
                // }

                /* In our case we simply save the ADDR together with its VAL in a 
                * hashmap. For now we will just handle the HEX format.
                * TODO: Support also binary hashmap
                */
                result.insert(
                    pos, n
                );
                //println!("pos = 0x{:x}, value = 0x{:x}", pos, n);
                // we save the entire HEX value in one address and have to increment by 4 bytes (32 word width)
                pos += 4; 
            }
        }
    }
    Ok(result)

}



pub fn hello() {
    println!("Hello, world from READMEM!");
}