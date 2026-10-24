use std::fmt;

use tritium_isa::{ImmediateOperation, Instruction, Register, RegisterOperation, Special};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembleError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for AssembleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for AssembleError {}

pub fn assemble(source: &str) -> Result<Vec<u32>, AssembleError> {
    let mut words = Vec::new();
    for (offset, raw_line) in source.lines().enumerate() {
        let text = raw_line.trim();
        if text.is_empty() {
            continue;
        }
        let (mnemonic, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        let operands: Vec<&str> = if rest.trim().is_empty() {
            Vec::new()
        } else {
            rest.split(',').map(str::trim).collect()
        };
        let instruction = encode(mnemonic, &operands)
            .map_err(|message| AssembleError { line: offset + 1, message })?;
        words.push(instruction.encode());
    }
    Ok(words)
}

fn encode(mnemonic: &str, operands: &[&str]) -> Result<Instruction, String> {
    match mnemonic {
        "lui" => {
            expect_operands(operands, 2)?;
            let value = parse_number(operands[1])?;
            let immediate = u16::try_from(value)
                .map_err(|_| format!("{value} does not fit in 16 unsigned bits"))?;
            return Ok(Instruction::LoadUpper {
                destination: parse_register(operands[0])?,
                immediate,
            });
        }
        "sreg" => {
            expect_operands(operands, 2)?;
            let special = Special::from_mnemonic(operands[1])
                .ok_or_else(|| format!("unknown special register {}", operands[1]))?;
            return Ok(Instruction::ReadSpecial {
                destination: parse_register(operands[0])?,
                special,
            });
        }
        "bar" => {
            expect_operands(operands, 0)?;
            return Ok(Instruction::Barrier);
        }
        "exit" => {
            expect_operands(operands, 0)?;
            return Ok(Instruction::Exit);
        }
        _ => {}
    }
    if let Some(operation) = RegisterOperation::from_mnemonic(mnemonic) {
        expect_operands(operands, 3)?;
        return Ok(Instruction::Register {
            operation,
            destination: parse_register(operands[0])?,
            source: parse_register(operands[1])?,
            second_source: parse_register(operands[2])?,
        });
    }
    if let Some(operation) = ImmediateOperation::from_mnemonic(mnemonic) {
        expect_operands(operands, 3)?;
        let is_logical = matches!(
            operation,
            ImmediateOperation::AndImmediate
                | ImmediateOperation::OrImmediate
                | ImmediateOperation::XorImmediate
        );
        let immediate =
            if is_logical { parse_logical_16(operands[2])? } else { parse_signed_16(operands[2])? };
        return Ok(Instruction::Immediate {
            operation,
            destination: parse_register(operands[0])?,
            source: parse_register(operands[1])?,
            immediate,
        });
    }
    Err(format!("unknown mnemonic {mnemonic}"))
}

fn expect_operands(operands: &[&str], count: usize) -> Result<(), String> {
    if operands.len() == count {
        Ok(())
    } else {
        Err(format!("expected {count} operands, found {}", operands.len()))
    }
}

fn parse_register(text: &str) -> Result<Register, String> {
    text.strip_prefix('r')
        .and_then(|digits| digits.parse::<u8>().ok())
        .and_then(Register::new)
        .ok_or_else(|| format!("unknown register {text}"))
}

fn parse_number(text: &str) -> Result<i64, String> {
    let (is_negative, magnitude) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let value = match magnitude.strip_prefix("0x").or_else(|| magnitude.strip_prefix("0X")) {
        Some(hex) => i64::from_str_radix(hex, 16),
        None => magnitude.parse::<i64>(),
    }
    .map_err(|_| format!("invalid number {text}"))?;
    Ok(if is_negative { -value } else { value })
}

fn parse_signed_16(text: &str) -> Result<i16, String> {
    let value = parse_number(text)?;
    i16::try_from(value).map_err(|_| format!("{value} does not fit in 16 signed bits"))
}

fn parse_logical_16(text: &str) -> Result<i16, String> {
    let value = parse_number(text)?;
    if (i64::from(i16::MIN)..=i64::from(u16::MAX)).contains(&value) {
        Ok(value as u16 as i16)
    } else {
        Err(format!("{value} does not fit in 16 bits"))
    }
}
