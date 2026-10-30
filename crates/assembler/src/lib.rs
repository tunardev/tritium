use std::collections::HashMap;
use std::fmt;

use tritium_isa::{
    BranchCondition, DecodeError, ImmediateOperation, Instruction, LoadWidth, MemorySpace,
    Register, RegisterOperation, Special, StoreWidth,
};

const SHARED_SUFFIX: &str = "s";
const JUMP_OFFSET_BITS: u32 = 21;

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

struct Statement<'a> {
    line: usize,
    mnemonic: &'a str,
    operands: Vec<&'a str>,
    index: usize,
}

pub fn assemble(source: &str) -> Result<Vec<u32>, AssembleError> {
    let mut labels = HashMap::new();
    let mut statements = Vec::new();
    let mut index = 0;
    for (offset, raw_line) in source.lines().enumerate() {
        let line = offset + 1;
        let text = raw_line.trim();
        if text.is_empty() {
            continue;
        }
        let error = |message: String| AssembleError { line, message };
        if let Some(label) = text.strip_suffix(':') {
            if !is_identifier(label) {
                return Err(error(format!("invalid label {label:?}")));
            }
            if labels.insert(label, index).is_some() {
                return Err(error(format!("label {label} is defined twice")));
            }
            continue;
        }
        let (mnemonic, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        let operands = if rest.trim().is_empty() {
            Vec::new()
        } else {
            rest.split(',').map(str::trim).collect()
        };
        let size = statement_size(mnemonic, &operands).map_err(error)?;
        statements.push(Statement { line, mnemonic, operands, index });
        index += size;
    }

    let mut words = Vec::with_capacity(index);
    for statement in &statements {
        let instructions = encode_statement(statement, &labels)
            .map_err(|message| AssembleError { line: statement.line, message })?;
        words.extend(instructions.iter().map(Instruction::encode));
    }
    Ok(words)
}

pub fn disassemble(words: &[u32]) -> Result<String, DecodeError> {
    let lines = words
        .iter()
        .map(|&word| Instruction::decode(word).map(|instruction| instruction.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(lines.join("\n"))
}

fn statement_size(mnemonic: &str, operands: &[&str]) -> Result<usize, String> {
    if mnemonic != "li" {
        return Ok(1);
    }
    let value = parse_load_immediate(operands.get(1).copied().unwrap_or_default())?;
    Ok(if i16::try_from(value).is_ok() { 1 } else { 2 })
}

fn encode_statement(
    statement: &Statement<'_>,
    labels: &HashMap<&str, usize>,
) -> Result<Vec<Instruction>, String> {
    let operands = &statement.operands;
    let mnemonic = statement.mnemonic;
    let target = |text: &str, bits: u32| resolve_target(text, statement.index, labels, bits);

    match mnemonic {
        "li" => {
            expect_operands(operands, 2)?;
            let destination = parse_register(operands[0])?;
            let value = parse_load_immediate(operands[1])?;
            if let Ok(immediate) = i16::try_from(value) {
                return Ok(vec![Instruction::Immediate {
                    operation: ImmediateOperation::AddImmediate,
                    destination,
                    source: Register::ZERO,
                    immediate,
                }]);
            }
            let bits = value as u32;
            Ok(vec![
                Instruction::LoadUpper { destination, immediate: (bits >> 16) as u16 },
                Instruction::Immediate {
                    operation: ImmediateOperation::OrImmediate,
                    destination,
                    source: destination,
                    immediate: bits as u16 as i16,
                },
            ])
        }
        "mov" => {
            expect_operands(operands, 2)?;
            Ok(vec![Instruction::Register {
                operation: RegisterOperation::Add,
                destination: parse_register(operands[0])?,
                source: parse_register(operands[1])?,
                second_source: Register::ZERO,
            }])
        }
        "j" => {
            expect_operands(operands, 1)?;
            Ok(vec![Instruction::JumpAndLink {
                destination: Register::ZERO,
                offset: target(operands[0], JUMP_OFFSET_BITS)? as i32,
            }])
        }
        "jal" => {
            expect_operands(operands, 2)?;
            Ok(vec![Instruction::JumpAndLink {
                destination: parse_register(operands[0])?,
                offset: target(operands[1], JUMP_OFFSET_BITS)? as i32,
            }])
        }
        "jalr" => {
            expect_operands(operands, 3)?;
            Ok(vec![Instruction::JumpAndLinkRegister {
                destination: parse_register(operands[0])?,
                base: parse_register(operands[1])?,
                offset: parse_signed_16(operands[2])?,
            }])
        }
        "lui" => {
            expect_operands(operands, 2)?;
            let value = parse_number(operands[1])?;
            let immediate = u16::try_from(value)
                .map_err(|_| format!("{value} does not fit in 16 unsigned bits"))?;
            Ok(vec![Instruction::LoadUpper {
                destination: parse_register(operands[0])?,
                immediate,
            }])
        }
        "sreg" => {
            expect_operands(operands, 2)?;
            let special = Special::from_mnemonic(operands[1])
                .ok_or_else(|| format!("unknown special register {}", operands[1]))?;
            Ok(vec![Instruction::ReadSpecial {
                destination: parse_register(operands[0])?,
                special,
            }])
        }
        "bar" => {
            expect_operands(operands, 0)?;
            Ok(vec![Instruction::Barrier])
        }
        "exit" => {
            expect_operands(operands, 0)?;
            Ok(vec![Instruction::Exit])
        }
        _ => encode_operation(mnemonic, operands, target).map(|instruction| vec![instruction]),
    }
}

fn encode_operation(
    mnemonic: &str,
    operands: &[&str],
    target: impl Fn(&str, u32) -> Result<i64, String>,
) -> Result<Instruction, String> {
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
        let is_shift = matches!(
            operation,
            ImmediateOperation::ShiftLeftImmediate
                | ImmediateOperation::ShiftRightLogicalImmediate
                | ImmediateOperation::ShiftRightArithmeticImmediate
        );
        let immediate = if is_logical {
            parse_unsigned_16(operands[2])?
        } else if is_shift {
            parse_shift_amount(operands[2])?
        } else {
            parse_signed_16(operands[2])?
        };
        return Ok(Instruction::Immediate {
            operation,
            destination: parse_register(operands[0])?,
            source: parse_register(operands[1])?,
            immediate,
        });
    }
    if let Some(condition) = BranchCondition::from_mnemonic(mnemonic) {
        expect_operands(operands, 3)?;
        return Ok(Instruction::Branch {
            condition,
            source: parse_register(operands[0])?,
            second_source: parse_register(operands[1])?,
            offset: target(operands[2], 16)? as i16,
        });
    }
    let (base_mnemonic, space) = match mnemonic.strip_suffix(SHARED_SUFFIX) {
        Some(stripped)
            if LoadWidth::from_mnemonic(stripped).is_some()
                || StoreWidth::from_mnemonic(stripped).is_some() =>
        {
            (stripped, MemorySpace::Shared)
        }
        _ => (mnemonic, MemorySpace::Global),
    };
    if let Some(width) = LoadWidth::from_mnemonic(base_mnemonic) {
        expect_operands(operands, 2)?;
        let (offset, base) = parse_memory(operands[1])?;
        return Ok(Instruction::Load {
            width,
            space,
            destination: parse_register(operands[0])?,
            base,
            offset,
        });
    }
    if let Some(width) = StoreWidth::from_mnemonic(base_mnemonic) {
        expect_operands(operands, 2)?;
        let (offset, base) = parse_memory(operands[1])?;
        return Ok(Instruction::Store {
            width,
            space,
            value: parse_register(operands[0])?,
            base,
            offset,
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

fn is_identifier(text: &str) -> bool {
    let mut characters = text.chars();
    characters.next().is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
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

fn parse_unsigned_16(text: &str) -> Result<i16, String> {
    let value = parse_number(text)?;
    u16::try_from(value)
        .map(|unsigned| unsigned as i16)
        .map_err(|_| format!("{value} does not fit in 16 unsigned bits"))
}

fn parse_shift_amount(text: &str) -> Result<i16, String> {
    let value = parse_number(text)?;
    if (0..32).contains(&value) {
        Ok(value as i16)
    } else {
        Err(format!("shift amount {value} is outside 0 to 31"))
    }
}

fn parse_load_immediate(text: &str) -> Result<i64, String> {
    let value = parse_number(text)?;
    if (i64::from(i32::MIN)..=i64::from(u32::MAX)).contains(&value) {
        Ok(if value > i64::from(i32::MAX) { i64::from(value as u32 as i32) } else { value })
    } else {
        Err(format!("{value} does not fit in 32 bits"))
    }
}

fn parse_memory(text: &str) -> Result<(i16, Register), String> {
    let (offset, rest) =
        text.split_once('(').ok_or_else(|| format!("expected offset(base), found {text}"))?;
    let base = rest.strip_suffix(')').ok_or_else(|| format!("expected ) in {text}"))?;
    let offset = if offset.trim().is_empty() { 0 } else { parse_signed_16(offset.trim())? };
    Ok((offset, parse_register(base.trim())?))
}

fn resolve_target(
    text: &str,
    index: usize,
    labels: &HashMap<&str, usize>,
    bits: u32,
) -> Result<i64, String> {
    let offset = match labels.get(text) {
        Some(&target) => target as i64 - index as i64,
        None if is_identifier(text) => return Err(format!("unknown label {text}")),
        None => parse_number(text)?,
    };
    let limit = 1i64 << (bits - 1);
    if (-limit..limit).contains(&offset) {
        Ok(offset)
    } else {
        Err(format!("target {text} is {offset} instructions away, beyond the {bits}-bit offset"))
    }
}
