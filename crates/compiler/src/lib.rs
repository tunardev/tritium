mod generator;
mod lexer;
mod parser;
mod syntax;

use std::fmt;

use tritium_assembler::assemble;
use tritium_isa::LaunchConfig;

pub use syntax::{Element, Parameter, Type};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl CompileError {
    fn new(line: usize, column: usize, message: impl Into<String>) -> CompileError {
        CompileError { line, column, message: message.into() }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for CompileError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub shared_bytes: u32,
    pub assembly: String,
    pub program: Vec<u32>,
}

impl Kernel {
    pub fn launch_config(&self, grid_blocks: u32, block_threads: u32) -> LaunchConfig {
        LaunchConfig { grid_blocks, block_threads, shared_bytes: self.shared_bytes }
    }
}

pub fn compile(source: &str) -> Result<Vec<Kernel>, CompileError> {
    let tokens = lexer::tokenize(source)?;
    let definitions = parser::Parser::new(tokens).parse_kernels()?;
    let mut kernels: Vec<Kernel> = Vec::new();
    for definition in definitions {
        if kernels.iter().any(|kernel| kernel.name == definition.name) {
            return Err(CompileError::new(
                definition.position.line,
                definition.position.column,
                format!("kernel {} is defined twice", definition.name),
            ));
        }
        let generated = generator::generate(&definition)?;
        let program = assemble(&generated.assembly).map_err(|error| {
            CompileError::new(
                definition.position.line,
                definition.position.column,
                format!("generated assembly failed to assemble: {error}"),
            )
        })?;
        kernels.push(Kernel {
            name: definition.name,
            parameters: definition.parameters,
            shared_bytes: generated.shared_bytes,
            assembly: generated.assembly,
            program,
        });
    }
    Ok(kernels)
}

pub fn compile_kernel(source: &str, name: &str) -> Result<Kernel, CompileError> {
    compile(source)?
        .into_iter()
        .find(|kernel| kernel.name == name)
        .ok_or_else(|| CompileError::new(1, 1, format!("no kernel named {name}")))
}
