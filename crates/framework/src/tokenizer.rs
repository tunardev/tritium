use std::collections::HashMap;
use std::fmt::Write;

pub const END_OF_TEXT: &str = "<|endoftext|>";
const BYTE_TOKENS: u32 = 256;
const HEADER: &str = "tritium-tokenizer 1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    pieces: Vec<Vec<u8>>,
}

fn chunks(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut previous_class = None;
    for (index, character) in text.char_indices() {
        let class = if character.is_alphabetic() {
            1
        } else if character.is_numeric() {
            2
        } else if character == ' ' {
            0
        } else {
            3
        };
        let continues = match previous_class {
            Some(0) => class == 1 || class == 2,
            Some(previous) => previous == class && class != 3,
            None => true,
        };
        if !continues && index > start {
            chunks.push(&text[start..index]);
            start = index;
        }
        previous_class = Some(class);
    }
    if start < text.len() {
        chunks.push(&text[start..]);
    }
    chunks
}

impl Tokenizer {
    pub fn train(text: &str, vocabulary_size: usize) -> Tokenizer {
        assert!(
            vocabulary_size > BYTE_TOKENS as usize + 1,
            "the vocabulary must exceed 257 tokens"
        );
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for story in text.split(END_OF_TEXT) {
            for chunk in chunks(story) {
                *counts.entry(chunk).or_default() += 1;
            }
        }
        let mut words: Vec<(Vec<u32>, usize)> = counts
            .into_iter()
            .map(|(chunk, count)| (chunk.bytes().map(u32::from).collect(), count))
            .collect();
        words.sort();
        let mut merges = Vec::new();
        let first_merged = BYTE_TOKENS + 1;
        while merges.len() + (first_merged as usize) < vocabulary_size {
            let mut pairs: HashMap<(u32, u32), usize> = HashMap::new();
            for (symbols, count) in &words {
                for pair in symbols.windows(2) {
                    *pairs.entry((pair[0], pair[1])).or_default() += count;
                }
            }
            let Some((&pair, _)) =
                pairs.iter().max_by_key(|&(&pair, &count)| (count, std::cmp::Reverse(pair)))
            else {
                break;
            };
            let token = first_merged + merges.len() as u32;
            merges.push(pair);
            for (symbols, _) in &mut words {
                *symbols = merge_pair(symbols, pair, token);
            }
        }
        Tokenizer::from_merges(merges)
    }

    fn from_merges(merges: Vec<(u32, u32)>) -> Tokenizer {
        let mut pieces: Vec<Vec<u8>> = (0..BYTE_TOKENS).map(|byte| vec![byte as u8]).collect();
        pieces.push(END_OF_TEXT.as_bytes().to_vec());
        let mut ranks = HashMap::new();
        for (rank, &(left, right)) in merges.iter().enumerate() {
            let mut piece = pieces[left as usize].clone();
            piece.extend_from_slice(&pieces[right as usize]);
            pieces.push(piece);
            ranks.insert((left, right), rank as u32);
        }
        Tokenizer { merges, ranks, pieces }
    }

    pub fn vocabulary_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn end_of_text(&self) -> u32 {
        BYTE_TOKENS
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
        for (index, story) in text.split(END_OF_TEXT).enumerate() {
            if index > 0 {
                tokens.push(self.end_of_text());
            }
            for chunk in chunks(story) {
                let encoded = cache.entry(chunk).or_insert_with(|| self.encode_chunk(chunk));
                tokens.extend_from_slice(encoded);
            }
        }
        tokens
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut symbols: Vec<u32> = chunk.bytes().map(u32::from).collect();
        loop {
            let best = symbols
                .windows(2)
                .filter_map(|pair| {
                    self.ranks.get(&(pair[0], pair[1])).map(|&rank| (rank, (pair[0], pair[1])))
                })
                .min();
            let Some((rank, pair)) = best else { return symbols };
            symbols = merge_pair(&symbols, pair, BYTE_TOKENS + 1 + rank);
        }
    }

    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        tokens.iter().flat_map(|&token| self.pieces[token as usize].iter().copied()).collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
    }

    pub fn to_text(&self) -> String {
        let mut text = format!("{HEADER}\n");
        for (left, right) in &self.merges {
            writeln!(text, "{left} {right}").unwrap();
        }
        text
    }

    pub fn from_text(text: &str) -> Result<Tokenizer, String> {
        let mut lines = text.lines();
        if lines.next() != Some(HEADER) {
            return Err("not a tritium tokenizer file".into());
        }
        let mut merges = Vec::new();
        for (index, line) in lines.enumerate() {
            let parse = |part: Option<&str>| {
                part.and_then(|part| part.parse::<u32>().ok())
                    .filter(|&token| token < BYTE_TOKENS + 1 + index as u32)
                    .ok_or_else(|| format!("line {} is not a valid merge", index + 2))
            };
            let mut parts = line.split(' ');
            merges.push((parse(parts.next())?, parse(parts.next())?));
        }
        Ok(Tokenizer::from_merges(merges))
    }
}

fn merge_pair(symbols: &[u32], pair: (u32, u32), token: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(symbols.len());
    let mut index = 0;
    while index < symbols.len() {
        if index + 1 < symbols.len() && (symbols[index], symbols[index + 1]) == pair {
            merged.push(token);
            index += 2;
        } else {
            merged.push(symbols[index]);
            index += 1;
        }
    }
    merged
}
