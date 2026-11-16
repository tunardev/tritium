use std::collections::{HashMap, HashSet};

use tritium_isa::{MAX_ARGUMENTS, SHARED_BYTES};

use crate::CompileError;
use crate::syntax::{
    BinaryOperator, Element, Expression, ExpressionKind, KernelDefinition, Place, Position,
    Statement, StatementKind, Type, UnaryOperator,
};

const FIRST_FREE_REGISTER: u8 = MAX_ARGUMENTS as u8 + 1;
const REGISTER_COUNT: u8 = 32;
const VARYING_BUILTINS: [&str; 2] = ["thread_index", "lane_index"];
const SPECIAL_BUILTINS: [&str; 6] =
    ["thread_index", "block_index", "block_threads", "grid_blocks", "lane_index", "warp_index"];

#[derive(Debug, Clone, Copy)]
enum Binding {
    Register { register: u8, value_type: Type },
    Shared { element: Element, offset: u32 },
}

#[derive(Debug, Clone, Copy)]
struct Value {
    register: u8,
    value_type: Type,
    is_temporary: bool,
}

pub struct Generated {
    pub assembly: String,
    pub shared_bytes: u32,
}

pub fn generate(kernel: &KernelDefinition) -> Result<Generated, CompileError> {
    Generator::new(kernel)?.run(kernel)
}

struct Generator {
    lines: Vec<String>,
    free: Vec<u8>,
    scopes: Vec<HashMap<String, Binding>>,
    varying: HashSet<String>,
    labels: usize,
    shared_bytes: u32,
}

fn error<T>(position: Position, message: impl Into<String>) -> Result<T, CompileError> {
    Err(CompileError::new(position.line, position.column, message))
}

impl Generator {
    fn new(kernel: &KernelDefinition) -> Result<Generator, CompileError> {
        if kernel.parameters.len() > MAX_ARGUMENTS {
            return error(
                kernel.position,
                format!("kernel {} has more than {MAX_ARGUMENTS} parameters", kernel.name),
            );
        }
        let mut scope = HashMap::new();
        for (index, parameter) in kernel.parameters.iter().enumerate() {
            let binding = Binding::Register {
                register: index as u8 + 1,
                value_type: parameter.parameter_type,
            };
            scope.insert(parameter.name.clone(), binding);
        }
        let mut shared_bytes = 0u32;
        for array in &kernel.shared {
            let offset = shared_bytes.next_multiple_of(4);
            shared_bytes = offset + array.element.size() * array.length;
            scope.insert(array.name.clone(), Binding::Shared { element: array.element, offset });
        }
        if shared_bytes as usize > SHARED_BYTES {
            return error(
                kernel.position,
                format!("shared arrays need {shared_bytes} bytes, more than {SHARED_BYTES}"),
            );
        }
        Ok(Generator {
            lines: Vec::new(),
            free: (FIRST_FREE_REGISTER..REGISTER_COUNT).rev().collect(),
            scopes: vec![scope],
            varying: varying_variables(&kernel.body),
            labels: 0,
            shared_bytes,
        })
    }

    fn run(mut self, kernel: &KernelDefinition) -> Result<Generated, CompileError> {
        self.statements(&kernel.body)?;
        self.emit("exit");
        Ok(Generated { assembly: self.lines.join("\n"), shared_bytes: self.shared_bytes })
    }

    fn emit(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    fn label(&mut self) -> String {
        self.labels += 1;
        format!("label_{}", self.labels)
    }

    fn allocate(&mut self, position: Position) -> Result<u8, CompileError> {
        self.free.pop().map_or_else(
            || error(position, "the kernel needs more than 23 registers at this point"),
            Ok,
        )
    }

    fn release(&mut self, value: Value) {
        if value.is_temporary {
            self.free.push(value.register);
        }
    }

    fn temporary(&mut self, position: Position, value_type: Type) -> Result<Value, CompileError> {
        Ok(Value { register: self.allocate(position)?, value_type, is_temporary: true })
    }

    fn lookup(&self, name: &str, position: Position) -> Result<Binding, CompileError> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
            .map_or_else(|| error(position, format!("unknown name {name}")), Ok)
    }

    fn statements(&mut self, statements: &[Statement]) -> Result<(), CompileError> {
        self.scopes.push(HashMap::new());
        for statement in statements {
            self.statement(statement)?;
        }
        let scope = self.scopes.pop().unwrap();
        for binding in scope.values() {
            if let Binding::Register { register, .. } = binding {
                self.free.push(*register);
            }
        }
        Ok(())
    }

    fn statement(&mut self, statement: &Statement) -> Result<(), CompileError> {
        let position = statement.position;
        match &statement.kind {
            StatementKind::Let(name, value) => {
                let value = self.expression(value)?;
                let register = if value.is_temporary {
                    value.register
                } else {
                    let register = self.allocate(position)?;
                    self.emit(format!("mov r{register}, r{}", value.register));
                    register
                };
                let binding = Binding::Register { register, value_type: value.value_type };
                self.scopes.last_mut().unwrap().insert(name.clone(), binding);
            }
            StatementKind::Assign(place, operator, value) => {
                let value = match operator {
                    Some(operator) => {
                        let current = match place {
                            Place::Variable(name) => ExpressionKind::Variable(name.clone()),
                            Place::Index(name, index) => {
                                ExpressionKind::Index(name.clone(), Box::new(index.clone()))
                            }
                        };
                        let current = Expression { kind: current, position };
                        self.binary(*operator, &current, value, position)?
                    }
                    None => self.expression(value)?,
                };
                self.assign(place, value, position)?;
            }
            StatementKind::If(condition, then_body, else_body) => {
                let (else_label, end_label) = (self.label(), self.label());
                let condition = self.condition(condition, "if")?;
                self.emit(format!("beq r{}, r0, {else_label}", condition.register));
                self.release(condition);
                self.statements(then_body)?;
                self.emit(format!("j {end_label}\n{else_label}:"));
                self.statements(else_body)?;
                self.emit(format!("{end_label}:"));
            }
            StatementKind::While(condition, body) => {
                let (start_label, end_label) = (self.label(), self.label());
                self.emit(format!("{start_label}:"));
                let condition = self.condition(condition, "while")?;
                self.emit(format!("beq r{}, r0, {end_label}", condition.register));
                self.release(condition);
                self.statements(body)?;
                self.emit(format!("j {start_label}\n{end_label}:"));
            }
            StatementKind::For(name, start, end, body) => self.for_loop(name, start, end, body)?,
            StatementKind::Expression(expression) => {
                if let ExpressionKind::Call(name, arguments) = &expression.kind {
                    if name == "barrier" {
                        if !arguments.is_empty() {
                            return error(position, "barrier takes no arguments");
                        }
                        self.emit("bar");
                        return Ok(());
                    }
                }
                let value = self.expression(expression)?;
                self.release(value);
            }
            StatementKind::Return => self.emit("exit"),
        }
        Ok(())
    }

    fn for_loop(
        &mut self,
        name: &str,
        start: &Expression,
        end: &Expression,
        body: &[Statement],
    ) -> Result<(), CompileError> {
        for bound in [start, end] {
            if self.is_varying(bound) {
                return error(
                    bound.position,
                    "loop bounds vary across lanes; branches must be uniform",
                );
            }
        }
        let start = self.expression(start)?;
        let counter = self.allocate(end.position)?;
        self.emit(format!("mov r{counter}, r{}", start.register));
        self.release(start);
        let end_value = self.expression(end)?;
        let limit = self.allocate(end.position)?;
        self.emit(format!("mov r{limit}, r{}", end_value.register));
        self.release(end_value);
        let (start_label, end_label) = (self.label(), self.label());
        self.emit(format!("{start_label}:\nbge r{counter}, r{limit}, {end_label}"));
        let mut scope = HashMap::new();
        scope.insert(
            name.to_string(),
            Binding::Register { register: counter, value_type: Type::Integer },
        );
        self.scopes.push(scope);
        self.statements(body)?;
        self.scopes.pop();
        self.emit(format!("addi r{counter}, r{counter}, 1\nj {start_label}\n{end_label}:"));
        self.free.push(limit);
        self.free.push(counter);
        Ok(())
    }

    fn condition(&mut self, condition: &Expression, keyword: &str) -> Result<Value, CompileError> {
        if self.is_varying(condition) {
            return error(
                condition.position,
                format!(
                    "the {keyword} condition varies across lanes; branches must be uniform, use select"
                ),
            );
        }
        self.expression(condition)
    }

    fn is_varying(&self, expression: &Expression) -> bool {
        is_varying(expression, &self.varying)
    }

    fn assign(
        &mut self,
        place: &Place,
        value: Value,
        position: Position,
    ) -> Result<(), CompileError> {
        match place {
            Place::Variable(name) => match self.lookup(name, position)? {
                Binding::Register { register, value_type } => {
                    if value_type != value.value_type {
                        return error(position, format!("{name} cannot change its type"));
                    }
                    self.emit(format!("mov r{register}, r{}", value.register));
                }
                Binding::Shared { .. } => {
                    return error(position, format!("{name} is a shared array; assign an element"));
                }
            },
            Place::Index(name, index) => {
                if value.value_type != Type::Integer {
                    return error(position, "only integers can be stored");
                }
                let (address, element, is_shared) = self.element_address(name, index, position)?;
                let mnemonic = match element.size() {
                    1 => "sb",
                    2 => "sh",
                    _ => "sw",
                };
                let suffix = if is_shared { "s" } else { "" };
                self.emit(format!(
                    "{mnemonic}{suffix} r{}, 0(r{})",
                    value.register, address.register
                ));
                self.release(address);
            }
        }
        self.release(value);
        Ok(())
    }

    fn element_address(
        &mut self,
        name: &str,
        index: &Expression,
        position: Position,
    ) -> Result<(Value, Element, bool), CompileError> {
        let index = self.expression(index)?;
        if index.value_type != Type::Integer {
            return error(position, "an index must be an integer");
        }
        let address = self.temporary(position, Type::Integer)?;
        let (element, is_shared) = match self.lookup(name, position)? {
            Binding::Register { register, value_type: Type::Pointer(element) } => {
                self.emit(format!(
                    "slli r{0}, r{1}, {2}\nadd r{0}, r{0}, r{register}",
                    address.register,
                    index.register,
                    element.shift()
                ));
                (element, false)
            }
            Binding::Shared { element, offset } => {
                self.emit(format!(
                    "slli r{0}, r{1}, {2}",
                    address.register,
                    index.register,
                    element.shift()
                ));
                if offset != 0 {
                    self.emit(format!("addi r{0}, r{0}, {offset}", address.register));
                }
                (element, true)
            }
            Binding::Register { .. } => return error(position, format!("{name} is not a pointer")),
        };
        self.release(index);
        Ok((address, element, is_shared))
    }

    fn expression(&mut self, expression: &Expression) -> Result<Value, CompileError> {
        let position = expression.position;
        match &expression.kind {
            ExpressionKind::Integer(value) => {
                if !(i64::from(i32::MIN)..=i64::from(u32::MAX)).contains(value) {
                    return error(position, format!("{value} does not fit in 32 bits"));
                }
                let result = self.temporary(position, Type::Integer)?;
                self.emit(format!("li r{}, {value}", result.register));
                Ok(result)
            }
            ExpressionKind::Variable(name) => match self.lookup(name, position)? {
                Binding::Register { register, value_type } => {
                    Ok(Value { register, value_type, is_temporary: false })
                }
                Binding::Shared { .. } => error(
                    position,
                    format!("{name} is a shared array; index it to read an element"),
                ),
            },
            ExpressionKind::Binary(operator, left, right) => {
                self.binary(*operator, left, right, position)
            }
            ExpressionKind::Unary(operator, operand) => {
                let operand = self.integer_operand(operand)?;
                self.release(operand);
                let result = self.temporary(position, Type::Integer)?;
                let (target, source) = (result.register, operand.register);
                self.emit(match operator {
                    UnaryOperator::Negate => format!("sub r{target}, r0, r{source}"),
                    UnaryOperator::Not => {
                        format!("sltu r{target}, r0, r{source}\nxori r{target}, r{target}, 1")
                    }
                });
                Ok(result)
            }
            ExpressionKind::Index(name, index) => {
                let (address, element, is_shared) = self.element_address(name, index, position)?;
                let mnemonic = match element {
                    Element::I8 => "lb",
                    Element::U8 => "lbu",
                    Element::I16 => "lh",
                    Element::U16 => "lhu",
                    Element::I32 => "lw",
                };
                let suffix = if is_shared { "s" } else { "" };
                self.emit(format!("{mnemonic}{suffix} r{0}, 0(r{0})", address.register));
                Ok(address)
            }
            ExpressionKind::Call(name, arguments) => self.call(name, arguments, position),
        }
    }

    fn integer_operand(&mut self, expression: &Expression) -> Result<Value, CompileError> {
        let value = self.expression(expression)?;
        if value.value_type != Type::Integer {
            return error(expression.position, "expected an integer, found a pointer");
        }
        Ok(value)
    }

    fn binary(
        &mut self,
        operator: BinaryOperator,
        left: &Expression,
        right: &Expression,
        position: Position,
    ) -> Result<Value, CompileError> {
        let left = self.expression(left)?;
        let right_value = self.expression(right)?;
        let result_type = match (operator, left.value_type, right_value.value_type) {
            (_, Type::Integer, Type::Integer) => Type::Integer,
            (
                BinaryOperator::Add | BinaryOperator::Subtract,
                Type::Pointer(element),
                Type::Integer,
            ) => Type::Pointer(element),
            (
                BinaryOperator::Equal | BinaryOperator::NotEqual | BinaryOperator::Less,
                Type::Pointer(_),
                Type::Pointer(_),
            ) => Type::Integer,
            _ => return error(position, "these operand types cannot be combined"),
        };
        let result = self.temporary(position, result_type)?;
        let (target, first, second) = (result.register, left.register, right_value.register);
        if let Type::Pointer(element) = result_type {
            let operation = if operator == BinaryOperator::Add { "add" } else { "sub" };
            self.emit(format!(
                "slli r{target}, r{second}, {}\n{operation} r{target}, r{first}, r{target}",
                element.shift()
            ));
            self.release(left);
            self.release(right_value);
            return Ok(result);
        }
        let operation = match operator {
            BinaryOperator::Add => format!("add r{target}, r{first}, r{second}"),
            BinaryOperator::Subtract => format!("sub r{target}, r{first}, r{second}"),
            BinaryOperator::Multiply => format!("mul16 r{target}, r{first}, r{second}"),
            BinaryOperator::And => format!("and r{target}, r{first}, r{second}"),
            BinaryOperator::Or => format!("or r{target}, r{first}, r{second}"),
            BinaryOperator::Xor => format!("xor r{target}, r{first}, r{second}"),
            BinaryOperator::ShiftLeft => format!("sll r{target}, r{first}, r{second}"),
            BinaryOperator::ShiftRight => format!("sra r{target}, r{first}, r{second}"),
            BinaryOperator::Less => format!("slt r{target}, r{first}, r{second}"),
            BinaryOperator::Greater => format!("slt r{target}, r{second}, r{first}"),
            BinaryOperator::LessOrEqual => {
                format!("slt r{target}, r{second}, r{first}\nxori r{target}, r{target}, 1")
            }
            BinaryOperator::GreaterOrEqual => {
                format!("slt r{target}, r{first}, r{second}\nxori r{target}, r{target}, 1")
            }
            BinaryOperator::Equal => format!(
                "xor r{target}, r{first}, r{second}\nsltu r{target}, r0, r{target}\n\
                 xori r{target}, r{target}, 1"
            ),
            BinaryOperator::NotEqual => {
                format!("xor r{target}, r{first}, r{second}\nsltu r{target}, r0, r{target}")
            }
            BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr => {
                let scratch = self.allocate(position)?;
                let combine = if operator == BinaryOperator::LogicalAnd { "and" } else { "or" };
                self.free.push(scratch);
                format!(
                    "sltu r{scratch}, r0, r{second}\nsltu r{target}, r0, r{first}\n\
                     {combine} r{target}, r{target}, r{scratch}"
                )
            }
        };
        self.emit(operation);
        self.release(left);
        self.release(right_value);
        Ok(result)
    }

    fn call(
        &mut self,
        name: &str,
        arguments: &[Expression],
        position: Position,
    ) -> Result<Value, CompileError> {
        let expect = |count: usize| {
            if arguments.len() == count {
                Ok(())
            } else {
                error(
                    position,
                    format!("{name} takes {count} arguments, found {}", arguments.len()),
                )
            }
        };
        if SPECIAL_BUILTINS.contains(&name) {
            expect(0)?;
            let result = self.temporary(position, Type::Integer)?;
            self.emit(format!("sreg r{}, {name}", result.register));
            return Ok(result);
        }
        match name {
            "min" | "max" | "scale" => {
                expect(2)?;
                self.register_operation(name, &arguments[0], Some(&arguments[1]), position)
            }
            "exp2" | "rsqrt" | "sat8" => {
                expect(1)?;
                self.register_operation(name, &arguments[0], None, position)
            }
            "select" => {
                expect(3)?;
                let condition = self.integer_operand(&arguments[0])?;
                let result = self.temporary(position, Type::Integer)?;
                self.emit(format!("mov r{}, r{}", result.register, condition.register));
                self.release(condition);
                let (chosen, other) =
                    (self.integer_operand(&arguments[1])?, self.integer_operand(&arguments[2])?);
                self.emit(format!(
                    "select r{}, r{}, r{}",
                    result.register, chosen.register, other.register
                ));
                self.release(chosen);
                self.release(other);
                Ok(result)
            }
            "tdot" | "dot8" => {
                expect(4)?;
                let accumulator = self.integer_operand(&arguments[0])?;
                let result = self.temporary(position, Type::Integer)?;
                self.emit(format!("mov r{}, r{}", result.register, accumulator.register));
                self.release(accumulator);
                let weights = self.expression(&arguments[1])?;
                if !matches!(weights.value_type, Type::Pointer(_)) {
                    return error(
                        arguments[1].position,
                        format!("{name} weights must be a pointer"),
                    );
                }
                let ExpressionKind::Variable(array) = &arguments[2].kind else {
                    return error(
                        arguments[2].position,
                        format!("{name} activations must be a shared i8 array"),
                    );
                };
                let (address, element, is_shared) =
                    self.element_address(array, &arguments[3], position)?;
                if !is_shared || element != Element::I8 {
                    return error(
                        arguments[2].position,
                        format!("{name} activations must be a shared i8 array"),
                    );
                }
                self.emit(format!(
                    "{name} r{}, r{}, r{}",
                    result.register, weights.register, address.register
                ));
                self.release(weights);
                self.release(address);
                Ok(result)
            }
            "barrier" => error(position, "barrier is a statement, not a value"),
            _ => error(position, format!("unknown function {name}")),
        }
    }

    fn register_operation(
        &mut self,
        mnemonic: &str,
        first: &Expression,
        second: Option<&Expression>,
        position: Position,
    ) -> Result<Value, CompileError> {
        let first = self.integer_operand(first)?;
        let second = second.map(|second| self.integer_operand(second)).transpose()?;
        self.release(first);
        if let Some(second) = second {
            self.release(second);
        }
        let result = self.temporary(position, Type::Integer)?;
        let second_register = second.map_or(0, |second| second.register);
        self.emit(format!(
            "{mnemonic} r{}, r{}, r{second_register}",
            result.register, first.register
        ));
        Ok(result)
    }
}

fn is_varying(expression: &Expression, varying: &HashSet<String>) -> bool {
    match &expression.kind {
        ExpressionKind::Integer(_) => false,
        ExpressionKind::Variable(name) => varying.contains(name),
        ExpressionKind::Binary(_, left, right) => {
            is_varying(left, varying) || is_varying(right, varying)
        }
        ExpressionKind::Unary(_, operand) => is_varying(operand, varying),
        ExpressionKind::Index(name, index) => varying.contains(name) || is_varying(index, varying),
        ExpressionKind::Call(name, arguments) => {
            VARYING_BUILTINS.contains(&name.as_str())
                || arguments.iter().any(|argument| is_varying(argument, varying))
        }
    }
}

fn varying_variables(body: &[Statement]) -> HashSet<String> {
    let mut varying = HashSet::new();
    loop {
        let before = varying.len();
        mark_varying(body, &mut varying);
        if varying.len() == before {
            return varying;
        }
    }
}

fn mark_varying(statements: &[Statement], varying: &mut HashSet<String>) {
    for statement in statements {
        match &statement.kind {
            StatementKind::Let(name, value)
            | StatementKind::Assign(Place::Variable(name), _, value) => {
                if is_varying(value, varying) {
                    varying.insert(name.clone());
                }
            }
            StatementKind::If(_, then_body, else_body) => {
                mark_varying(then_body, varying);
                mark_varying(else_body, varying);
            }
            StatementKind::While(_, body) => mark_varying(body, varying),
            StatementKind::For(name, start, _, body) => {
                if is_varying(start, varying) {
                    varying.insert(name.clone());
                }
                mark_varying(body, varying);
            }
            StatementKind::Assign(Place::Index(..), ..)
            | StatementKind::Expression(_)
            | StatementKind::Return => {}
        }
    }
}
