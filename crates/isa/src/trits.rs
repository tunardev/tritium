pub const TRITS_PER_BYTE: usize = 4;
pub const TRITS_PER_WORD: usize = 4 * TRITS_PER_BYTE;

fn trit_code(trit: i8) -> u8 {
    match trit {
        0 => 0,
        1 => 1,
        -1 => 2,
        _ => panic!("{trit} is not a trit"),
    }
}

fn code_trit(code: u8) -> i8 {
    match code {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

pub fn pack_trits(trits: &[i8]) -> Vec<u8> {
    let padded_length = trits.len().div_ceil(TRITS_PER_WORD).max(1) * TRITS_PER_WORD;
    let mut padded = trits.to_vec();
    padded.resize(padded_length, 0);
    padded
        .chunks_exact(TRITS_PER_BYTE)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u8, |byte, (index, &trit)| byte | (trit_code(trit) << (2 * index)))
        })
        .collect()
}

pub fn unpack_byte(byte: u8) -> [i8; TRITS_PER_BYTE] {
    let codes: [u8; TRITS_PER_BYTE] = std::array::from_fn(|index| (byte >> (2 * index)) & 3);
    if codes.contains(&3) {
        return [0; TRITS_PER_BYTE];
    }
    codes.map(code_trit)
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
    fn every_valid_byte_round_trips() {
        for byte in 0u8..=255 {
            let is_valid = (0..TRITS_PER_BYTE).all(|index| (byte >> (2 * index)) & 3 != 3);
            if is_valid {
                assert_eq!(pack_trits(&unpack_byte(byte))[0], byte);
            }
        }
    }

    #[test]
    fn invalid_codes_decode_to_zero() {
        assert_eq!(unpack_byte(0xFF), [0; TRITS_PER_BYTE]);
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
