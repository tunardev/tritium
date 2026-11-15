pub const TRITS_PER_BYTE: usize = 5;
pub const TRITS_PER_WORD: usize = 4 * TRITS_PER_BYTE;
pub const INT8_PER_WORD: usize = 4;

const VALID_BYTES: u8 = 243;

fn trit_digit(trit: i8) -> u8 {
    match trit {
        -1..=1 => (trit + 1) as u8,
        _ => panic!("{trit} is not a trit"),
    }
}

pub fn pack_trits(trits: &[i8]) -> Vec<u8> {
    let padded_length = trits.len().div_ceil(TRITS_PER_WORD).max(1) * TRITS_PER_WORD;
    let mut padded = trits.to_vec();
    padded.resize(padded_length, 0);
    padded
        .chunks_exact(TRITS_PER_BYTE)
        .map(|chunk| chunk.iter().rev().fold(0u8, |byte, &trit| byte * 3 + trit_digit(trit)))
        .collect()
}

pub fn unpack_byte(byte: u8) -> [i8; TRITS_PER_BYTE] {
    if byte >= VALID_BYTES {
        return [0; TRITS_PER_BYTE];
    }
    let mut remaining = byte;
    std::array::from_fn(|_| {
        let digit = remaining % 3;
        remaining /= 3;
        digit as i8 - 1
    })
}

pub fn unpack_word(word: u32) -> [i8; TRITS_PER_WORD] {
    let mut trits = [0i8; TRITS_PER_WORD];
    for (index, byte) in word.to_le_bytes().into_iter().enumerate() {
        trits[index * TRITS_PER_BYTE..][..TRITS_PER_BYTE].copy_from_slice(&unpack_byte(byte));
    }
    trits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_weights_use_1_6_bits() {
        let trits = vec![1i8; 2000];
        assert!(pack_trits(&trits).len() * 8 <= trits.len() * 16 / 10);
    }

    #[test]
    fn every_valid_byte_round_trips() {
        for byte in 0u8..243 {
            assert_eq!(pack_trits(&unpack_byte(byte))[0], byte);
        }
    }

    #[test]
    fn invalid_bytes_decode_to_zero() {
        for byte in 243u8..=255 {
            assert_eq!(unpack_byte(byte), [0; TRITS_PER_BYTE]);
        }
    }

    #[test]
    fn packing_pads_rows_to_words() {
        let packed = pack_trits(&[1, -1, 0, 1, 1, -1, 1]);
        assert_eq!(packed.len(), 4);
        let word = u32::from_le_bytes(packed.try_into().unwrap());
        assert_eq!(&unpack_word(word)[..7], &[1, -1, 0, 1, 1, -1, 1]);
        assert!(unpack_word(word)[7..].iter().all(|&trit| trit == 0));
    }

    #[test]
    #[should_panic(expected = "not a trit")]
    fn values_outside_trits_are_rejected() {
        pack_trits(&[2]);
    }
}
