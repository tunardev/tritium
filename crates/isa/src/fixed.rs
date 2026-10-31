use std::sync::OnceLock;

const ONE: i64 = 1 << 16;
const EXP2_TABLE_SIZE: usize = 256;
const RSQRT_TABLE_SIZE: usize = 192;

pub fn exp2_table() -> [i32; EXP2_TABLE_SIZE] {
    static TABLE: OnceLock<[i32; EXP2_TABLE_SIZE]> = OnceLock::new();
    *TABLE.get_or_init(|| {
        std::array::from_fn(|index| {
            (2f64.powf(index as f64 / EXP2_TABLE_SIZE as f64) * ONE as f64).round() as i32
        })
    })
}

pub fn rsqrt_table() -> [i32; RSQRT_TABLE_SIZE] {
    static TABLE: OnceLock<[i32; RSQRT_TABLE_SIZE]> = OnceLock::new();
    *TABLE.get_or_init(|| {
        std::array::from_fn(|index| {
            let mantissa = (index + 64) as f64 / 64.0;
            (ONE as f64 / mantissa.sqrt()).round() as i32
        })
    })
}

fn saturate_to_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

pub fn exp2(x: i32) -> i32 {
    if x >= 0 {
        return ONE as i32;
    }
    let integer = x >> 16;
    let fraction = (x & 0xFFFF) as usize;
    let shift = -integer;
    if shift > 16 {
        return 0;
    }
    let value = exp2_table()[(fraction * EXP2_TABLE_SIZE) >> 16];
    (value + (1 << (shift - 1))) >> shift
}

pub fn rsqrt(x: i32) -> i32 {
    let value = x as u32;
    if value == 0 {
        return i32::MAX;
    }
    let exponent = 31 - value.leading_zeros() as i32;
    let even_exponent = exponent & !1;
    let mantissa = if even_exponent >= 6 {
        value >> (even_exponent - 6)
    } else {
        value << (6 - even_exponent)
    };
    let estimate = i64::from(rsqrt_table()[mantissa as usize - 64]);
    if even_exponent <= 16 {
        saturate_to_i32(estimate << ((16 - even_exponent) / 2))
    } else {
        let shift = (even_exponent - 16) / 2;
        ((estimate + (1 << (shift - 1))) >> shift) as i32
    }
}

pub fn scale(value: i32, factor: i32) -> i32 {
    saturate_to_i32((i64::from(value) * i64::from(factor) + (1 << 15)) >> 16)
}

pub fn saturate_to_int8(value: i32) -> i32 {
    value.clamp(-127, 127)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp2_matches_known_points() {
        assert_eq!(exp2(0), 65536);
        assert_eq!(exp2(100), 65536);
        assert_eq!(exp2(-65536), 32768);
        assert_eq!(exp2(-17 * 65536), 0);
    }

    #[test]
    fn exp2_stays_within_table_resolution() {
        for x in (-16 * 65536..0).step_by(997) {
            let exact = 2f64.powf(x as f64 / 65536.0) * 65536.0;
            assert!((exp2(x) as f64 - exact).abs() <= exact * 0.003 + 1.0, "{x}");
        }
    }

    #[test]
    fn rsqrt_matches_known_points() {
        assert_eq!(rsqrt(65536), 65536);
        assert_eq!(rsqrt(4 * 65536), 32768);
        assert_eq!(rsqrt(65536 / 4), 131072);
        assert_eq!(rsqrt(0), i32::MAX);
    }

    #[test]
    fn rsqrt_stays_within_table_resolution() {
        for x in (1..i32::MAX).step_by(7_340_033) {
            let exact = 65536.0 / (x as f64 / 65536.0).sqrt();
            let actual = rsqrt(x) as f64;
            assert!(
                (actual - exact).abs() <= exact * 0.006 + 1.0 || actual == i32::MAX as f64,
                "{x}"
            );
        }
    }

    #[test]
    fn scale_rounds_and_saturates() {
        assert_eq!(scale(3, 32768), 2);
        assert_eq!(scale(-3, 32768), -1);
        assert_eq!(scale(i32::MAX, i32::MAX), i32::MAX);
        assert_eq!(saturate_to_int8(300), 127);
        assert_eq!(saturate_to_int8(-300), -127);
    }
}
