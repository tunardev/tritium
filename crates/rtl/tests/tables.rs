use std::fs;
use std::path::PathBuf;

use tritium_isa::unpack_byte;

fn hardware_file(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../hardware/rtl").join(name)
}

fn assert_file_matches(name: &str, expected: &str) {
    let path = hardware_file(name);
    if std::env::var_os("UPDATE_EXPECT").is_some() {
        fs::write(&path, expected).unwrap();
    }
    let actual = fs::read_to_string(&path).unwrap_or_default();
    assert!(actual == expected, "{name} is stale; rerun with UPDATE_EXPECT=1");
}

fn trit_bits(trit: i8) -> &'static str {
    match trit {
        -1 => "11",
        0 => "00",
        _ => "01",
    }
}

#[test]
fn trit_table_matches_isa_unpacking() {
    let mut text = String::from(
        "function automatic logic [9:0] trit_digits(input logic [7:0] value);\n  \
         unique case (value)\n",
    );
    for byte in 0..=255u8 {
        let bits: String = unpack_byte(byte).iter().rev().map(|&trit| trit_bits(trit)).collect();
        text += &format!("    8'd{byte}: trit_digits = 10'b{bits};\n");
    }
    text += "    default: trit_digits = '0;\n  endcase\nendfunction\n";
    assert_file_matches("trit_table.svh", &text);
}
