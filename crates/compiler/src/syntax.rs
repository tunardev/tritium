#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    I8,
    U8,
    I16,
    U16,
    I32,
}

impl Element {
    pub fn from_name(name: &str) -> Option<Element> {
        match name {
            "i8" => Some(Element::I8),
            "u8" => Some(Element::U8),
            "i16" => Some(Element::I16),
            "u16" => Some(Element::U16),
            "i32" => Some(Element::I32),
            _ => None,
        }
    }

    pub fn size(self) -> u32 {
        match self {
            Element::I8 | Element::U8 => 1,
            Element::I16 | Element::U16 => 2,
            Element::I32 => 4,
        }
    }

    pub fn shift(self) -> u32 {
        self.size().trailing_zeros()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Integer,
    Pointer(Element),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Integer(i64),
    Variable(String),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    Unary(UnaryOperator, Box<Expression>),
    Call(String, Vec<Expression>),
    Index(String, Box<Expression>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub position: Position,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Place {
    Variable(String),
    Index(String, Expression),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementKind {
    Let(String, Expression),
    Assign(Place, Option<BinaryOperator>, Expression),
    If(Expression, Vec<Statement>, Vec<Statement>),
    While(Expression, Vec<Statement>),
    For(String, Expression, Expression, Vec<Statement>),
    Expression(Expression),
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statement {
    pub kind: StatementKind,
    pub position: Position,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub name: String,
    pub parameter_type: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedArray {
    pub name: String,
    pub element: Element,
    pub length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelDefinition {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub shared: Vec<SharedArray>,
    pub body: Vec<Statement>,
    pub position: Position,
}
