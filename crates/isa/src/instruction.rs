use std::fmt;

pub const WARP_LANES: usize = 16;
pub const SHARED_BYTES: usize = 16384;
pub const MAX_ARGUMENTS: usize = 8;
pub const REGISTER_COUNT: usize = 32;

const REGISTER_OPCODE: u32 = 0x00;
const LOAD_UPPER_OPCODE: u32 = 0x09;
const READ_SPECIAL_OPCODE: u32 = 0x0A;
const JUMP_AND_LINK_REGISTER_OPCODE: u32 = 0x0B;
const JUMP_AND_LINK_OPCODE: u32 = 0x0C;
const BARRIER_OPCODE: u32 = 0x30;
const EXIT_OPCODE: u32 = 0x31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Register(u8);

impl Register {
    pub const ZERO: Register = Register(0);

    pub fn new(index: u8) -> Option<Register> {
        (usize::from(index) < REGISTER_COUNT).then_some(Register(index))
    }

    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    fn field(self, shift: u32) -> u32 {
        u32::from(self.0) << shift
    }
}

impl fmt::Display for Register {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "r{}", self.0)
    }
}

macro_rules! operation_enum {
    ($name:ident { $($variant:ident = $code:expr, $mnemonic:expr;)* }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $($variant,)*
        }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant,)*];

            pub fn code(self) -> u32 {
                match self {
                    $($name::$variant => $code,)*
                }
            }

            pub fn mnemonic(self) -> &'static str {
                match self {
                    $($name::$variant => $mnemonic,)*
                }
            }

            pub fn from_code(code: u32) -> Option<$name> {
                $name::ALL.iter().copied().find(|operation| operation.code() == code)
            }

            pub fn from_mnemonic(mnemonic: &str) -> Option<$name> {
                $name::ALL.iter().copied().find(|operation| operation.mnemonic() == mnemonic)
            }
        }
    };
}

operation_enum!(RegisterOperation {
    Add = 0, "add";
    Subtract = 1, "sub";
    And = 2, "and";
    Or = 3, "or";
    Xor = 4, "xor";
    ShiftLeft = 5, "sll";
    ShiftRightLogical = 6, "srl";
    ShiftRightArithmetic = 7, "sra";
    SetLessThan = 8, "slt";
    SetLessThanUnsigned = 9, "sltu";
    Minimum = 10, "min";
    Maximum = 11, "max";
    MultiplyBytes = 12, "mul8";
    Select = 13, "select";
});

operation_enum!(ImmediateOperation {
    AddImmediate = 0x01, "addi";
    AndImmediate = 0x02, "andi";
    OrImmediate = 0x03, "ori";
    XorImmediate = 0x04, "xori";
    ShiftLeftImmediate = 0x05, "slli";
    ShiftRightLogicalImmediate = 0x06, "srli";
    ShiftRightArithmeticImmediate = 0x07, "srai";
    SetLessThanImmediate = 0x08, "slti";
});

operation_enum!(LoadWidth {
    Word = 0, "lw";
    Half = 1, "lh";
    HalfUnsigned = 2, "lhu";
    Byte = 3, "lb";
    ByteUnsigned = 4, "lbu";
});

operation_enum!(StoreWidth {
    Word = 0, "sw";
    Half = 1, "sh";
    Byte = 2, "sb";
});

operation_enum!(BranchCondition {
    Equal = 0x20, "beq";
    NotEqual = 0x21, "bne";
    LessThan = 0x22, "blt";
    GreaterOrEqual = 0x23, "bge";
});

operation_enum!(Special {
    ThreadIndex = 0, "thread_index";
    BlockIndex = 1, "block_index";
    BlockThreads = 2, "block_threads";
    GridBlocks = 3, "grid_blocks";
    LaneIndex = 4, "lane_index";
    WarpIndex = 5, "warp_index";
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Global,
    Shared,
}

impl MemorySpace {
    fn load_opcode_base(self) -> u32 {
        match self {
            MemorySpace::Global => 0x10,
            MemorySpace::Shared => 0x15,
        }
    }

    fn store_opcode_base(self) -> u32 {
        match self {
            MemorySpace::Global => 0x1A,
            MemorySpace::Shared => 0x1D,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            MemorySpace::Global => "",
            MemorySpace::Shared => "s",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    Register {
        operation: RegisterOperation,
        destination: Register,
        source: Register,
        second_source: Register,
    },
    Immediate {
        operation: ImmediateOperation,
        destination: Register,
        source: Register,
        immediate: i16,
    },
    Load {
        width: LoadWidth,
        space: MemorySpace,
        destination: Register,
        base: Register,
        offset: i16,
    },
    Store {
        width: StoreWidth,
        space: MemorySpace,
        value: Register,
        base: Register,
        offset: i16,
    },
    Branch {
        condition: BranchCondition,
        source: Register,
        second_source: Register,
        offset: i16,
    },
    LoadUpper {
        destination: Register,
        immediate: u16,
    },
    ReadSpecial {
        destination: Register,
        special: Special,
    },
    JumpAndLink {
        destination: Register,
        offset: i32,
    },
    JumpAndLinkRegister {
        destination: Register,
        base: Register,
        offset: i16,
    },
    Barrier,
    Exit,
}

fn opcode(code: u32) -> u32 {
    code << 26
}

fn immediate_field(value: i16) -> u32 {
    u32::from(value as u16)
}

impl Instruction {
    pub fn encode(&self) -> u32 {
        match *self {
            Instruction::Register { operation, destination, source, second_source } => {
                opcode(REGISTER_OPCODE)
                    | destination.field(21)
                    | source.field(16)
                    | second_source.field(11)
                    | operation.code()
            }
            Instruction::Immediate { operation, destination, source, immediate } => {
                opcode(operation.code())
                    | destination.field(21)
                    | source.field(16)
                    | immediate_field(immediate)
            }
            Instruction::Load { width, space, destination, base, offset } => {
                opcode(space.load_opcode_base() + width.code())
                    | destination.field(21)
                    | base.field(16)
                    | immediate_field(offset)
            }
            Instruction::Store { width, space, value, base, offset } => {
                opcode(space.store_opcode_base() + width.code())
                    | value.field(21)
                    | base.field(16)
                    | immediate_field(offset)
            }
            Instruction::Branch { condition, source, second_source, offset } => {
                opcode(condition.code())
                    | source.field(21)
                    | second_source.field(16)
                    | immediate_field(offset)
            }
            Instruction::LoadUpper { destination, immediate } => {
                opcode(LOAD_UPPER_OPCODE) | destination.field(21) | u32::from(immediate)
            }
            Instruction::ReadSpecial { destination, special } => {
                opcode(READ_SPECIAL_OPCODE) | destination.field(21) | special.code()
            }
            Instruction::JumpAndLink { destination, offset } => {
                opcode(JUMP_AND_LINK_OPCODE) | destination.field(21) | (offset as u32 & 0x1F_FFFF)
            }
            Instruction::JumpAndLinkRegister { destination, base, offset } => {
                opcode(JUMP_AND_LINK_REGISTER_OPCODE)
                    | destination.field(21)
                    | base.field(16)
                    | immediate_field(offset)
            }
            Instruction::Barrier => opcode(BARRIER_OPCODE),
            Instruction::Exit => opcode(EXIT_OPCODE),
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Instruction::Register { operation, destination, source, second_source } => {
                write!(
                    formatter,
                    "{} {destination}, {source}, {second_source}",
                    operation.mnemonic()
                )
            }
            Instruction::Immediate { operation, destination, source, immediate } => {
                write!(formatter, "{} {destination}, {source}, {immediate}", operation.mnemonic())
            }
            Instruction::Load { width, space, destination, base, offset } => write!(
                formatter,
                "{}{} {destination}, {offset}({base})",
                width.mnemonic(),
                space.suffix()
            ),
            Instruction::Store { width, space, value, base, offset } => write!(
                formatter,
                "{}{} {value}, {offset}({base})",
                width.mnemonic(),
                space.suffix()
            ),
            Instruction::Branch { condition, source, second_source, offset } => {
                write!(formatter, "{} {source}, {second_source}, {offset}", condition.mnemonic())
            }
            Instruction::LoadUpper { destination, immediate } => {
                write!(formatter, "lui {destination}, {immediate}")
            }
            Instruction::ReadSpecial { destination, special } => {
                write!(formatter, "sreg {destination}, {}", special.mnemonic())
            }
            Instruction::JumpAndLink { destination, offset } => {
                write!(formatter, "jal {destination}, {offset}")
            }
            Instruction::JumpAndLinkRegister { destination, base, offset } => {
                write!(formatter, "jalr {destination}, {base}, {offset}")
            }
            Instruction::Barrier => formatter.write_str("bar"),
            Instruction::Exit => formatter.write_str("exit"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register(index: u8) -> Register {
        Register::new(index).unwrap()
    }

    #[test]
    fn registers_beyond_31_are_invalid() {
        assert!(Register::new(31).is_some());
        assert!(Register::new(32).is_none());
    }

    #[test]
    fn instructions_print_in_assembler_syntax() {
        let load = Instruction::Load {
            width: LoadWidth::Word,
            space: MemorySpace::Shared,
            destination: register(4),
            base: register(5),
            offset: 8,
        };
        let add = Instruction::Register {
            operation: RegisterOperation::Add,
            destination: register(1),
            source: register(2),
            second_source: register(3),
        };
        assert_eq!(load.to_string(), "lws r4, 8(r5)");
        assert_eq!(add.to_string(), "add r1, r2, r3");
        assert_eq!(
            Instruction::ReadSpecial { destination: register(7), special: Special::LaneIndex }
                .to_string(),
            "sreg r7, lane_index"
        );
    }
}
