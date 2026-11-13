use crate::CompileError;
use crate::lexer::{Token, TokenKind};
use crate::syntax::{
    BinaryOperator, Element, Expression, ExpressionKind, KernelDefinition, Parameter, Place,
    Position, SharedArray, Statement, StatementKind, Type, UnaryOperator,
};

const PRECEDENCE: [&[(&str, BinaryOperator)]; 7] = [
    &[("||", BinaryOperator::LogicalOr)],
    &[("&&", BinaryOperator::LogicalAnd)],
    &[
        ("==", BinaryOperator::Equal),
        ("!=", BinaryOperator::NotEqual),
        ("<", BinaryOperator::Less),
        ("<=", BinaryOperator::LessOrEqual),
        (">", BinaryOperator::Greater),
        (">=", BinaryOperator::GreaterOrEqual),
    ],
    &[("|", BinaryOperator::Or), ("^", BinaryOperator::Xor)],
    &[("&", BinaryOperator::And)],
    &[("<<", BinaryOperator::ShiftLeft), (">>", BinaryOperator::ShiftRight)],
    &[("+", BinaryOperator::Add), ("-", BinaryOperator::Subtract)],
];

const COMPOUND_ASSIGNMENTS: [(&str, BinaryOperator); 6] = [
    ("+=", BinaryOperator::Add),
    ("-=", BinaryOperator::Subtract),
    ("*=", BinaryOperator::Multiply),
    ("&=", BinaryOperator::And),
    ("|=", BinaryOperator::Or),
    ("^=", BinaryOperator::Xor),
];

pub struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Parser {
        Parser { tokens, index: 0 }
    }

    pub fn parse_kernels(&mut self) -> Result<Vec<KernelDefinition>, CompileError> {
        let mut kernels = Vec::new();
        while self.peek().kind != TokenKind::End {
            kernels.push(self.parse_kernel()?);
        }
        Ok(kernels)
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.index]
    }

    fn position(&self) -> Position {
        let token = self.peek();
        Position { line: token.line, column: token.column }
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T, CompileError> {
        let token = self.peek();
        Err(CompileError::new(token.line, token.column, message))
    }

    fn advance(&mut self) -> Token {
        let token = self.tokens[self.index].clone();
        if token.kind != TokenKind::End {
            self.index += 1;
        }
        token
    }

    fn is_symbol(&self, symbol: &str) -> bool {
        matches!(self.peek().kind, TokenKind::Symbol(found) if found == symbol)
    }

    fn is_keyword(&self, keyword: &str) -> bool {
        matches!(&self.peek().kind, TokenKind::Identifier(found) if found == keyword)
    }

    fn accept(&mut self, symbol: &str) -> bool {
        let is_match = self.is_symbol(symbol);
        if is_match {
            self.advance();
        }
        is_match
    }

    fn expect(&mut self, symbol: &str) -> Result<(), CompileError> {
        if self.accept(symbol) { Ok(()) } else { self.error(format!("expected {symbol}")) }
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<(), CompileError> {
        if self.is_keyword(keyword) {
            self.advance();
            Ok(())
        } else {
            self.error(format!("expected {keyword}"))
        }
    }

    fn identifier(&mut self) -> Result<String, CompileError> {
        match self.peek().kind.clone() {
            TokenKind::Identifier(name) => {
                self.advance();
                Ok(name)
            }
            _ => self.error("expected a name"),
        }
    }

    fn integer(&mut self) -> Result<i64, CompileError> {
        match self.peek().kind {
            TokenKind::Integer(value) => {
                self.advance();
                Ok(value)
            }
            _ => self.error("expected an integer"),
        }
    }

    fn element(&mut self) -> Result<Element, CompileError> {
        let name = self.identifier()?;
        Element::from_name(&name).map_or_else(
            || self.error(format!("unknown element type {name}; expected i8, u8, i16, u16 or i32")),
            Ok,
        )
    }

    fn parse_type(&mut self) -> Result<Type, CompileError> {
        if self.accept("*") {
            return Ok(Type::Pointer(self.element()?));
        }
        let name = self.identifier()?;
        if name == "i32" { Ok(Type::Integer) } else { self.error(format!("unknown type {name}")) }
    }

    fn parse_kernel(&mut self) -> Result<KernelDefinition, CompileError> {
        let position = self.position();
        self.expect_keyword("kernel")?;
        let name = self.identifier()?;
        self.expect("(")?;
        let mut parameters = Vec::new();
        while !self.accept(")") {
            if !parameters.is_empty() {
                self.expect(",")?;
            }
            let name = self.identifier()?;
            self.expect(":")?;
            parameters.push(Parameter { name, parameter_type: self.parse_type()? });
        }
        self.expect("{")?;
        let mut shared = Vec::new();
        while self.is_keyword("shared") {
            self.advance();
            let name = self.identifier()?;
            self.expect(":")?;
            self.expect("[")?;
            let element = self.element()?;
            self.expect(";")?;
            let length = self.integer()?;
            self.expect("]")?;
            self.expect(";")?;
            let length = u32::try_from(length)
                .ok()
                .filter(|&length| length > 0)
                .map_or_else(|| self.error("shared array length must be positive"), Ok)?;
            shared.push(SharedArray { name, element, length });
        }
        let body = self.parse_statements()?;
        Ok(KernelDefinition { name, parameters, shared, body, position })
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>, CompileError> {
        self.expect("{")?;
        self.parse_statements()
    }

    fn parse_statements(&mut self) -> Result<Vec<Statement>, CompileError> {
        let mut statements = Vec::new();
        while !self.accept("}") {
            if self.peek().kind == TokenKind::End {
                return self.error("expected }");
            }
            statements.push(self.parse_statement()?);
        }
        Ok(statements)
    }

    fn parse_statement(&mut self) -> Result<Statement, CompileError> {
        let position = self.position();
        let kind = if self.is_keyword("let") {
            self.advance();
            let name = self.identifier()?;
            self.expect("=")?;
            let value = self.parse_expression()?;
            self.expect(";")?;
            StatementKind::Let(name, value)
        } else if self.is_keyword("if") {
            self.advance();
            let condition = self.parse_expression()?;
            let then_body = self.parse_block()?;
            let else_body = if self.is_keyword("else") {
                self.advance();
                if self.is_keyword("if") {
                    vec![self.parse_statement()?]
                } else {
                    self.parse_block()?
                }
            } else {
                Vec::new()
            };
            StatementKind::If(condition, then_body, else_body)
        } else if self.is_keyword("while") {
            self.advance();
            let condition = self.parse_expression()?;
            StatementKind::While(condition, self.parse_block()?)
        } else if self.is_keyword("for") {
            self.advance();
            let name = self.identifier()?;
            self.expect_keyword("in")?;
            let start = self.parse_expression()?;
            self.expect("..")?;
            let end = self.parse_expression()?;
            StatementKind::For(name, start, end, self.parse_block()?)
        } else if self.is_keyword("return") {
            self.advance();
            self.expect(";")?;
            StatementKind::Return
        } else {
            let expression = self.parse_expression()?;
            let place = match &expression.kind {
                ExpressionKind::Variable(name) => Some(Place::Variable(name.clone())),
                ExpressionKind::Index(name, index) => {
                    Some(Place::Index(name.clone(), *index.clone()))
                }
                _ => None,
            };
            let operator = if self.accept("=") {
                Some(None)
            } else {
                COMPOUND_ASSIGNMENTS
                    .iter()
                    .find(|(symbol, _)| self.is_symbol(symbol))
                    .map(|&(_, operator)| Some(operator))
            };
            let kind = match (place, operator) {
                (Some(place), Some(operator)) => {
                    if operator.is_some() {
                        self.advance();
                    }
                    StatementKind::Assign(place, operator, self.parse_expression()?)
                }
                (None, Some(_)) => {
                    return self.error("only variables and elements can be assigned");
                }
                (_, None) => StatementKind::Expression(expression),
            };
            self.expect(";")?;
            kind
        };
        Ok(Statement { kind, position })
    }

    pub fn parse_expression(&mut self) -> Result<Expression, CompileError> {
        self.parse_binary(0)
    }

    fn parse_binary(&mut self, level: usize) -> Result<Expression, CompileError> {
        if level == PRECEDENCE.len() {
            return self.parse_product();
        }
        let mut left = self.parse_binary(level + 1)?;
        loop {
            let position = self.position();
            let Some(&(_, operator)) =
                PRECEDENCE[level].iter().find(|(symbol, _)| self.is_symbol(symbol))
            else {
                return Ok(left);
            };
            self.advance();
            let right = self.parse_binary(level + 1)?;
            left = Expression {
                kind: ExpressionKind::Binary(operator, Box::new(left), Box::new(right)),
                position,
            };
        }
    }

    fn parse_product(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_unary()?;
        while self.is_symbol("*") {
            let position = self.position();
            self.advance();
            let right = self.parse_unary()?;
            left = Expression {
                kind: ExpressionKind::Binary(
                    BinaryOperator::Multiply,
                    Box::new(left),
                    Box::new(right),
                ),
                position,
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, CompileError> {
        let position = self.position();
        let operator = if self.accept("-") {
            UnaryOperator::Negate
        } else if self.accept("!") {
            UnaryOperator::Not
        } else {
            return self.parse_primary();
        };
        let operand = self.parse_unary()?;
        if let (UnaryOperator::Negate, ExpressionKind::Integer(value)) = (operator, &operand.kind) {
            return Ok(Expression { kind: ExpressionKind::Integer(-value), position });
        }
        Ok(Expression { kind: ExpressionKind::Unary(operator, Box::new(operand)), position })
    }

    fn parse_primary(&mut self) -> Result<Expression, CompileError> {
        let position = self.position();
        if self.accept("(") {
            let expression = self.parse_expression()?;
            self.expect(")")?;
            return Ok(expression);
        }
        if let TokenKind::Integer(value) = self.peek().kind {
            self.advance();
            return Ok(Expression { kind: ExpressionKind::Integer(value), position });
        }
        let name = self.identifier()?;
        let kind = if self.accept("(") {
            let mut arguments = Vec::new();
            while !self.accept(")") {
                if !arguments.is_empty() {
                    self.expect(",")?;
                }
                arguments.push(self.parse_expression()?);
            }
            ExpressionKind::Call(name, arguments)
        } else if self.accept("[") {
            let index = self.parse_expression()?;
            self.expect("]")?;
            ExpressionKind::Index(name, Box::new(index))
        } else {
            ExpressionKind::Variable(name)
        };
        Ok(Expression { kind, position })
    }
}
