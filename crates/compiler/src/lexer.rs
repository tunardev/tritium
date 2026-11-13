use crate::CompileError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Identifier(String),
    Integer(i64),
    Symbol(&'static str),
    End,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
}

const SYMBOLS: [&str; 34] = [
    "..", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "&=", "|=", "^=", "+",
    "-", "*", "&", "|", "^", "<", ">", "=", "!", "(", ")", "{", "}", "[", "]", ",", ";", ":",
];

pub fn tokenize(source: &str) -> Result<Vec<Token>, CompileError> {
    let mut tokens = Vec::new();
    for (line_index, line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let text = line.split("//").next().unwrap_or_default();
        let characters: Vec<char> = text.chars().collect();
        let mut index = 0;
        while index < characters.len() {
            let character = characters[index];
            let column = index + 1;
            if character.is_whitespace() {
                index += 1;
            } else if character.is_ascii_alphabetic() || character == '_' {
                let start = index;
                while index < characters.len()
                    && (characters[index].is_ascii_alphanumeric() || characters[index] == '_')
                {
                    index += 1;
                }
                let name: String = characters[start..index].iter().collect();
                tokens.push(Token { kind: TokenKind::Identifier(name), line: line_number, column });
            } else if character.is_ascii_digit() {
                let start = index;
                while index < characters.len()
                    && (characters[index].is_ascii_alphanumeric() || characters[index] == '_')
                {
                    index += 1;
                }
                let digits: String =
                    characters[start..index].iter().filter(|&&digit| digit != '_').collect();
                let value = match digits.strip_prefix("0x") {
                    Some(hex) => i64::from_str_radix(hex, 16),
                    None => digits.parse(),
                }
                .map_err(|_| {
                    CompileError::new(line_number, column, format!("invalid number {digits}"))
                })?;
                tokens.push(Token { kind: TokenKind::Integer(value), line: line_number, column });
            } else {
                let rest: String = characters[index..].iter().take(2).collect();
                let symbol =
                    SYMBOLS.iter().find(|symbol| rest.starts_with(**symbol)).ok_or_else(|| {
                        CompileError::new(
                            line_number,
                            column,
                            format!("unexpected character {character:?}"),
                        )
                    })?;
                index += symbol.len();
                tokens.push(Token { kind: TokenKind::Symbol(symbol), line: line_number, column });
            }
        }
    }
    let line = source.lines().count().max(1);
    tokens.push(Token { kind: TokenKind::End, line, column: 1 });
    Ok(tokens)
}
