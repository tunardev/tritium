use std::fs;
use std::io;
use std::path::Path;

use crate::random::Random;
use crate::tokenizer::Tokenizer;

pub struct TokenDataset {
    tokens: Vec<u16>,
}

impl TokenDataset {
    pub fn encode_file(tokenizer: &Tokenizer, text: &Path, output: &Path) -> io::Result<usize> {
        assert!(tokenizer.vocabulary_size() <= 1 << 16, "token ids must fit in 16 bits");
        let text = fs::read_to_string(text)?;
        let tokens = tokenizer.encode(&text);
        let bytes: Vec<u8> =
            tokens.iter().flat_map(|&token| (token as u16).to_le_bytes()).collect();
        fs::write(output, bytes)?;
        Ok(tokens.len())
    }

    pub fn open(path: &Path) -> io::Result<TokenDataset> {
        let bytes = fs::read(path)?;
        let tokens =
            bytes.chunks_exact(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]])).collect();
        Ok(TokenDataset { tokens })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn batch(
        &self,
        batch: usize,
        sequence: usize,
        random: &mut Random,
    ) -> (Vec<usize>, Vec<usize>) {
        assert!(self.tokens.len() > sequence + 1, "the dataset is shorter than one sequence");
        let mut inputs = Vec::with_capacity(batch * sequence);
        let mut targets = Vec::with_capacity(batch * sequence);
        for _ in 0..batch {
            let start = random.below(self.tokens.len() - sequence - 1);
            let window = &self.tokens[start..start + sequence + 1];
            inputs.extend(window[..sequence].iter().map(|&token| usize::from(token)));
            targets.extend(window[1..].iter().map(|&token| usize::from(token)));
        }
        (inputs, targets)
    }
}
