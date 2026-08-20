//! PTX Parser
//!
//! A Pratt parser for PTX constant expressions, following the style of
//! https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
//!
//! Extended to parse full PTX modules including directives, declarations,
//! functions, and instructions.

use crate::ascii::{AsciiChar, AsciiSliceExt, AsciiString, ascii};
use crate::ast::*;
use crate::instr::InstrKind;
use crate::lex::{DottedIdent, FloatLit, Ident, Lexer, Token};
use volta_common::report::Locate;

// =============================================================================
// Parse Error
// =============================================================================

/// The inner error type describing what went wrong (without location info).
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    UnexpectedEof,
    UnexpectedToken(Token),
    ExpectedToken {
        expected: Token,
        found: Option<Token>,
    },
    ExpectedDirective {
        expected: &'static [AsciiChar],
        found: Option<Token>,
    },
    InvalidDirective(AsciiString),
    InvalidModifier(AsciiString),
    InvalidType(AsciiString),
    InvalidStateSpace(AsciiString),
    InvalidStateSpaceQualifier(AsciiString),
    InvalidPtrSpace(AsciiString),
    ExpectedStateSpace(Option<Token>),
    ExpectedType(Option<Token>),
    ExpectedScalarType(Option<Token>),
    ExpectedIdentifier(Option<Token>),
    ExpectedInteger(Option<Token>),
    ExpectedPositiveInteger(i64),
    InvalidAlignment(u64),
    ExpectedOperand(Option<Token>),
    ExpectedInstruction(Option<Token>),
    ExpectedVersion(Option<Token>),
    ExpectedAddressSize(Option<Token>),
    ExpectedAddressBase(Option<Token>),
    ExpectedInitializer(Option<Token>),
    ExpectedFilename(Option<Token>),
    UnexpectedIdentAtStatement(AsciiString),
    EmptyQualifiedIdent,
    InvalidTargetOption(AsciiString),
    ConflictingTexmode,
    LexerError(crate::lex::Error),
}

/// Parse error with location information (path and span).
pub type ParseError = Locate<ParseErrorKind>;

impl ParseErrorKind {
    pub fn title(&self) -> &'static str {
        match self {
            ParseErrorKind::UnexpectedEof => "Unexpected End of File",
            ParseErrorKind::UnexpectedToken(_) => "Unexpected Token",
            ParseErrorKind::ExpectedToken { .. } => "Expected Token",
            ParseErrorKind::ExpectedDirective { .. } => "Expected Directive",
            ParseErrorKind::InvalidDirective(_) => "Invalid Directive",
            ParseErrorKind::InvalidModifier(_) => "Invalid Modifier",
            ParseErrorKind::InvalidType(_) => "Invalid Type",
            ParseErrorKind::InvalidStateSpace(_) => "Invalid State Space",
            ParseErrorKind::InvalidStateSpaceQualifier(_) => "Invalid State Space Qualifier",
            ParseErrorKind::InvalidPtrSpace(_) => "Invalid Pointer Space",
            ParseErrorKind::ExpectedStateSpace(_) => "Expected State Space",
            ParseErrorKind::ExpectedType(_) => "Expected Type",
            ParseErrorKind::ExpectedScalarType(_) => "Expected Scalar Type",
            ParseErrorKind::ExpectedIdentifier(_) => "Expected Identifier",
            ParseErrorKind::ExpectedInteger(_) => "Expected Integer",
            ParseErrorKind::ExpectedPositiveInteger(_) => "Expected Positive Integer",
            ParseErrorKind::InvalidAlignment(_) => "Invalid Alignment",
            ParseErrorKind::ExpectedOperand(_) => "Expected Operand",
            ParseErrorKind::ExpectedInstruction(_) => "Expected Instruction",
            ParseErrorKind::ExpectedVersion(_) => "Expected Version",
            ParseErrorKind::ExpectedAddressSize(_) => "Expected Address Size",
            ParseErrorKind::ExpectedAddressBase(_) => "Expected Address Base",
            ParseErrorKind::ExpectedInitializer(_) => "Expected Initializer",
            ParseErrorKind::ExpectedFilename(_) => "Expected Filename",
            ParseErrorKind::UnexpectedIdentAtStatement(_) => "Unexpected Identifier",
            ParseErrorKind::EmptyQualifiedIdent => "Empty Qualified Identifier",
            ParseErrorKind::InvalidTargetOption(_) => "Invalid Target Option",
            ParseErrorKind::ConflictingTexmode => "Conflicting Texture Modes",
            ParseErrorKind::LexerError(_) => "Lexer Error",
        }
    }

    pub fn message(&self) -> Option<String> {
        Some(match self {
            ParseErrorKind::UnexpectedEof => "Unexpected end of file while parsing.".to_string(),
            ParseErrorKind::UnexpectedToken(tok) => format!("Unexpected token: {}", tok),
            ParseErrorKind::ExpectedToken { expected, found } => match found {
                Some(tok) => format!("Expected {}, found {}", expected, tok),
                None => format!("Expected {}, found end of file", expected),
            },
            ParseErrorKind::ExpectedDirective { expected, found } => {
                let expected_str = expected
                    .iter()
                    .map(|c| *c as u8 as char)
                    .collect::<String>();
                match found {
                    Some(tok) => format!("Expected .{}, found {}", expected_str, tok),
                    None => format!("Expected .{}, found end of file", expected_str),
                }
            }
            ParseErrorKind::InvalidDirective(name) => format!("Invalid directive: .{}", name),
            ParseErrorKind::InvalidModifier(name) => format!("Invalid modifier: .{}", name),
            ParseErrorKind::InvalidType(name) => format!("Invalid type: .{}", name),
            ParseErrorKind::InvalidStateSpace(name) => format!("Invalid state space: .{}", name),
            ParseErrorKind::InvalidStateSpaceQualifier(name) => {
                format!("Invalid state space qualifier: .{}", name)
            }
            ParseErrorKind::InvalidPtrSpace(name) => format!(
                "Invalid pointer space: .{} (expected one of .const, .global, .local, .shared)",
                name
            ),
            ParseErrorKind::ExpectedStateSpace(found) => match found {
                Some(tok) => format!("Expected state space, found {}", tok),
                None => "Expected state space, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedType(found) => match found {
                Some(tok) => format!("Expected type, found {}", tok),
                None => "Expected type, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedScalarType(found) => match found {
                Some(tok) => format!("Expected scalar type, found {}", tok),
                None => "Expected scalar type, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedIdentifier(found) => match found {
                Some(tok) => format!("Expected identifier, found {}", tok),
                None => "Expected identifier, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedInteger(found) => match found {
                Some(tok) => format!("Expected integer, found {}", tok),
                None => "Expected integer, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedPositiveInteger(val) => {
                format!("Expected positive integer, got {}", val)
            }
            ParseErrorKind::InvalidAlignment(val) => {
                format!(".align requires a power of two, got {}", val)
            }
            ParseErrorKind::ExpectedOperand(found) => match found {
                Some(tok) => format!("Expected operand, found {}", tok),
                None => "Expected operand, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedInstruction(found) => match found {
                Some(tok) => format!("Expected instruction, found {}", tok),
                None => "Expected instruction, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedVersion(found) => match found {
                Some(tok) => format!("Expected version number, found {}", tok),
                None => "Expected version number, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedAddressSize(found) => match found {
                Some(tok) => format!("Expected 32 or 64 for address_size, found {}", tok),
                None => "Expected 32 or 64 for address_size, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedAddressBase(found) => match found {
                Some(tok) => format!("Expected address base, found {}", tok),
                None => "Expected address base, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedInitializer(found) => match found {
                Some(tok) => format!("Expected initializer value, found {}", tok),
                None => "Expected initializer value, found end of file".to_string(),
            },
            ParseErrorKind::ExpectedFilename(found) => match found {
                Some(tok) => format!("Expected filename string, found {}", tok),
                None => "Expected filename string, found end of file".to_string(),
            },
            ParseErrorKind::UnexpectedIdentAtStatement(name) => {
                format!("Unexpected identifier at statement level: {}", name)
            }
            ParseErrorKind::EmptyQualifiedIdent => "Empty qualified identifier.".to_string(),
            ParseErrorKind::InvalidTargetOption(name) => {
                format!("Invalid target option: {}", name)
            }
            ParseErrorKind::ConflictingTexmode => {
                "Conflicting texture modes specified.".to_string()
            }
            ParseErrorKind::LexerError(e) => format!("Invalid character at position {}", e.0),
        })
    }
}

/// Helper to create a located error at a specific span.
fn err_at<T>(span: Span, kind: ParseErrorKind) -> Result<T, ParseError> {
    Err(Locate {
        path: None,
        span: Some(span),
        error: kind,
    })
}

/// Helper to create a located error without a span.
fn err<T>(kind: ParseErrorKind) -> Result<T, ParseError> {
    Err(Locate {
        path: None,
        span: None,
        error: kind,
    })
}

// =============================================================================
// Binding Power
// =============================================================================

// Binding power constants (higher = tighter binding)
const BP_TERNARY: u8 = 3; // ?: (right-assoc, so l=4, r=3)
const BP_OR: u8 = 5; // ||
const BP_AND: u8 = 7; // &&
const BP_BIT_OR: u8 = 9; // |
const BP_BIT_XOR: u8 = 11; // ^
const BP_BIT_AND: u8 = 13; // &
const BP_EQ: u8 = 15; // == !=
const BP_CMP: u8 = 17; // < <= > >=
const BP_SHIFT: u8 = 19; // << >>
const BP_ADD: u8 = 21; // + -
const BP_MUL: u8 = 23; // * / %
const BP_UNARY: u8 = 27; // unary + - ! ~ and casts

/// Returns the binding power for prefix (unary) operators.
/// Returns ((), right_bp) since prefix operators only bind to the right.
fn prefix_binding_power(op: &Token) -> Option<((), u8)> {
    match op {
        Token::Plus | Token::Minus | Token::Bang | Token::Tilde => Some(((), BP_UNARY)),
        _ => None,
    }
}

/// Returns the binding power for infix (binary) operators.
/// Returns (left_bp, right_bp).
/// - Left-associative operators: (bp, bp+1)
/// - Right-associative operators: (bp+1, bp)
fn infix_binding_power(op: &Token) -> Option<(u8, u8)> {
    let lhs = |bp: u8| Some((bp, bp + 1));
    let rhs = |bp: u8| Some((bp + 1, bp));
    match op {
        Token::Star | Token::Slash | Token::Percent => lhs(BP_MUL),
        Token::Plus | Token::Minus => lhs(BP_ADD),
        Token::LeftShift | Token::RightShift => lhs(BP_SHIFT),
        Token::Less | Token::LessEquals | Token::Greater | Token::GreaterEquals => lhs(BP_CMP),
        Token::EqualsEquals | Token::BangEquals => lhs(BP_EQ),
        Token::Ampersand => lhs(BP_BIT_AND),
        Token::Caret => lhs(BP_BIT_XOR),
        Token::Pipe => lhs(BP_BIT_OR),
        Token::AmpersandAmpersand => lhs(BP_AND),
        Token::PipePipe => lhs(BP_OR),
        Token::Question => rhs(BP_TERNARY),
        _ => None,
    }
}

fn token_to_unary_op(tok: &Token) -> UnaryOp {
    match tok {
        Token::Plus => UnaryOp::Pos,
        Token::Minus => UnaryOp::Neg,
        Token::Bang => UnaryOp::Not,
        Token::Tilde => UnaryOp::BitNot,
        _ => unreachable!("not a unary operator: {:?}", tok),
    }
}

fn token_to_binary_op(tok: &Token) -> BinaryOp {
    match tok {
        Token::Plus => BinaryOp::Add,
        Token::Minus => BinaryOp::Sub,
        Token::Star => BinaryOp::Mul,
        Token::Slash => BinaryOp::Div,
        Token::Percent => BinaryOp::Rem,
        Token::LeftShift => BinaryOp::Shl,
        Token::RightShift => BinaryOp::Shr,
        Token::Less => BinaryOp::Lt,
        Token::LessEquals => BinaryOp::Le,
        Token::Greater => BinaryOp::Gt,
        Token::GreaterEquals => BinaryOp::Ge,
        Token::EqualsEquals => BinaryOp::Eq,
        Token::BangEquals => BinaryOp::Ne,
        Token::Ampersand => BinaryOp::BitAnd,
        Token::Caret => BinaryOp::BitXor,
        Token::Pipe => BinaryOp::BitOr,
        Token::AmpersandAmpersand => BinaryOp::And,
        Token::PipePipe => BinaryOp::Or,
        _ => unreachable!("not a binary operator: {:?}", tok),
    }
}

// =============================================================================
// Parser
// =============================================================================

/// Convert suffix qualifiers from Ident::as_instr() into DottedIdent modifiers.
/// E.g., for "st.async.weak" with mnemonic "st.async", the suffix is ".weak",
/// which splits to ["", "weak"]. This function filters empty parts and wraps
/// each in DottedIdent::Simple; a part containing `::` (e.g. `L2::128B` from
/// `ld.global.L2::128B.f32`) becomes DottedIdent::Qualified with the same
/// component split the lexer gives stand-alone dotted qualifiers.
fn parse_suffix_modifiers<'a>(suffix: impl Iterator<Item = &'a [AsciiChar]>) -> Vec<DottedIdent> {
    suffix
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut parts = split_double_colon(s);
            if parts.len() == 1 {
                DottedIdent::Simple(parts.pop().unwrap())
            } else {
                DottedIdent::Qualified(parts)
            }
        })
        .collect()
}

/// Split a modifier segment at each `::`. The lexer only ever consumes a
/// `::` when a component follows it, so the resulting parts are non-empty.
fn split_double_colon(s: &[AsciiChar]) -> Vec<AsciiString> {
    let mut parts = Vec::new();
    let mut rest = s;
    loop {
        match rest.windows(2).position(|w| w == ascii("::")) {
            Some(i) => {
                parts.push(rest[..i].to_owned_ascii());
                rest = &rest[i + 2..];
            }
            None => {
                parts.push(rest.to_owned_ascii());
                return parts;
            }
        }
    }
}

type LexerError = crate::lex::Error;

impl From<LexerError> for ParseError {
    fn from(e: LexerError) -> Self {
        Locate {
            path: None,
            span: Some(Span(e.0, e.0 + 1)),
            error: ParseErrorKind::LexerError(e),
        }
    }
}

pub struct Parser<'a> {
    lexer: std::iter::Peekable<Lexer<'a>>,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a [AsciiChar]) -> Self {
        Self {
            lexer: Lexer::new(src).peekable(),
        }
    }

    /// Peek at the current token and its span without consuming it.
    fn peek(&mut self) -> Result<Option<(Span, &Token)>, LexerError> {
        match self.lexer.peek() {
            Some(Ok((start, tok, end))) => Ok(Some((Span(*start, *end), tok))),
            Some(Err(e)) => Err(*e),
            None => Ok(None),
        }
    }

    /// Peek at just the token without span (convenience for comparisons).
    fn peek_tok(&mut self) -> Result<Option<&Token>, LexerError> {
        Ok(self.peek()?.map(|(_, tok)| tok))
    }

    /// Consume and return the current token with its span.
    fn next(&mut self) -> Result<Option<(Span, Token)>, LexerError> {
        match self.lexer.next() {
            Some(Ok((start, tok, end))) => Ok(Some((Span(start, end), tok))),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    /// Expect a specific token and consume it.
    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        match self.next()? {
            Some((_, tok)) if tok == expected => Ok(()),
            Some((span, other)) => err_at(
                span,
                ParseErrorKind::ExpectedToken {
                    expected,
                    found: Some(other),
                },
            ),
            None => err(ParseErrorKind::ExpectedToken {
                expected,
                found: None,
            }),
        }
    }

    /// Check if there's more input.
    fn is_eof(&mut self) -> Result<bool, ParseError> {
        Ok(self.peek_tok()?.is_none())
    }

    /// Expect an identifier and return it.
    fn expect_ident(&mut self) -> Result<Ident, ParseError> {
        match self.next()? {
            Some((_, Token::Ident(ident))) => Ok(ident),
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedIdentifier(Some(tok))),
            None => err(ParseErrorKind::ExpectedIdentifier(None)),
        }
    }

    /// Expect an identifier and return it with its span.
    fn parse_ident_with_span(&mut self) -> Result<(Span, Ident), ParseError> {
        match self.next()? {
            Some((span, Token::Ident(ident))) => Ok((span, ident)),
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedIdentifier(Some(tok))),
            None => err(ParseErrorKind::ExpectedIdentifier(None)),
        }
    }

    /// Expect a specific dotted identifier (directive name).
    fn expect_directive(&mut self, name: &'static [AsciiChar]) -> Result<(), ParseError> {
        match self.next()? {
            Some((_, Token::DottedIdent(DottedIdent::Simple(s)))) if s.as_slice() == name => Ok(()),
            Some((span, tok)) => err_at(
                span,
                ParseErrorKind::ExpectedDirective {
                    expected: name,
                    found: Some(tok),
                },
            ),
            None => err(ParseErrorKind::ExpectedDirective {
                expected: name,
                found: None,
            }),
        }
    }

    /// Try to consume a specific dotted identifier, return true if consumed.
    fn try_directive(&mut self, name: &[AsciiChar]) -> Result<bool, ParseError> {
        match self.peek()? {
            Some((_, Token::DottedIdent(DottedIdent::Simple(s)))) if s.as_slice() == name => {
                self.next().unwrap();
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Parse an integer literal (no expressions allowed - for array bounds/alignment).
    fn parse_int_literal(&mut self) -> Result<u64, ParseError> {
        match self.next()? {
            Some((span, Token::SIntLit(n))) => {
                if n.value() < 0 {
                    err_at(span, ParseErrorKind::ExpectedPositiveInteger(n.value()))
                } else {
                    Ok(n.value() as u64)
                }
            }
            Some((_, Token::UIntLit(n))) => Ok(n.value()),
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedInteger(Some(tok))),
            None => err(ParseErrorKind::ExpectedInteger(None)),
        }
    }

    // =========================================================================
    // Module-level parsing
    // =========================================================================

    /// Parse a complete PTX module.
    pub fn parse_module(&mut self) -> Result<Module, ParseError> {
        let version = self.parse_version()?;
        let target = self.parse_target()?;
        let address_size = self.parse_address_size()?;

        let mut items = Vec::new();
        while !self.is_eof()? {
            items.push(self.parse_top_level_item()?);
        }

        Ok(Module {
            version,
            target,
            address_size,
            items,
        })
    }

    /// Parse `.version major.minor`
    fn parse_version(&mut self) -> Result<Version, ParseError> {
        self.expect_directive(ascii("version"))?;

        match self.next()? {
            // Version is parsed as a float literal like `7.0` or `8.5`
            Some((span, ref tok @ Token::FloatLit(FloatLit::Dec(ref f)))) => {
                let err = || Locate {
                    path: None,
                    span: Some(span),
                    error: ParseErrorKind::ExpectedVersion(Some(tok.clone())),
                };
                if let Some(major) = f.characteristic()
                    && let Some(minor) = f.mantissa()
                    && let None = f.exponent()
                {
                    let major: u32 = major.as_str().parse().map_err(|_| err())?;
                    let minor: u32 = minor.as_str().parse().map_err(|_| err())?;
                    Ok(Version { major, minor })
                } else {
                    Err(err())
                }
            }
            // Version is parsed as an integer like `7` (no minor version)
            Some((span, ref tok @ Token::SIntLit(ref n))) => {
                let major: u32 = n.value().try_into().map_err(|_| Locate {
                    path: None,
                    span: Some(span),
                    error: ParseErrorKind::ExpectedVersion(Some(tok.clone())),
                })?;
                Ok(Version { major, minor: 0 })
            }
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedVersion(Some(tok))),
            None => err(ParseErrorKind::ExpectedVersion(None)),
        }
    }

    /// Parse `.target option[, option, ...]`
    /// Options can be: arch (sm_XX/compute_XX), texmode_unified, texmode_independent, debug, map_f64_to_f32
    /// Options can appear in any order. At least one option is required.
    fn parse_target(&mut self) -> Result<Target, ParseError> {
        self.expect_directive(ascii("target"))?;

        let mut archs: Vec<Arch> = Vec::new();
        let mut texmode: Option<Texmode> = None;
        let mut debug = false;
        let mut map_f64_to_f32 = false;

        // Parse first option (required)
        let (first_span, first) = self.parse_ident_with_span()?;
        Self::process_target_option(
            first_span,
            &first,
            &mut archs,
            &mut texmode,
            &mut debug,
            &mut map_f64_to_f32,
        )?;

        // Parse remaining comma-separated options
        while self.peek_tok()? == Some(&Token::Comma) {
            self.next().unwrap(); // consume comma
            let (opt_span, opt) = self.parse_ident_with_span()?;
            Self::process_target_option(
                opt_span,
                &opt,
                &mut archs,
                &mut texmode,
                &mut debug,
                &mut map_f64_to_f32,
            )?;
        }

        Ok(Target {
            archs,
            texmode,
            debug,
            map_f64_to_f32,
        })
    }

    /// Process a single target option, updating the appropriate field.
    fn process_target_option(
        span: Span,
        opt: &Ident,
        archs: &mut Vec<Arch>,
        texmode: &mut Option<Texmode>,
        debug: &mut bool,
        map_f64_to_f32: &mut bool,
    ) -> Result<(), ParseError> {
        if let Some(a) = Arch::from_ascii(opt.raw()) {
            archs.push(a);
        } else if let Some(t) = Texmode::from_ascii(opt.raw()) {
            if texmode.is_some() && *texmode != Some(t) {
                return err(ParseErrorKind::ConflictingTexmode);
            }
            *texmode = Some(t);
        } else if opt.raw() == ascii("debug") {
            *debug = true;
        } else if opt.raw() == ascii("map_f64_to_f32") {
            *map_f64_to_f32 = true;
        } else {
            return err_at(
                span,
                ParseErrorKind::InvalidTargetOption(opt.raw().to_owned_ascii()),
            );
        }
        Ok(())
    }

    /// Parse `.address_size 32|64` (optional directive)
    fn parse_address_size(&mut self) -> Result<Option<AddressSize>, ParseError> {
        if !self.try_directive(ascii("address_size"))? {
            return Ok(None);
        }

        match self.next()? {
            Some((_, Token::SIntLit(n))) if n.value() == 32 => Ok(Some(AddressSize::Bits32)),
            Some((_, Token::SIntLit(n))) if n.value() == 64 => Ok(Some(AddressSize::Bits64)),
            Some((_, Token::UIntLit(n))) if n.value() == 32 => Ok(Some(AddressSize::Bits32)),
            Some((_, Token::UIntLit(n))) if n.value() == 64 => Ok(Some(AddressSize::Bits64)),
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedAddressSize(Some(tok))),
            None => err(ParseErrorKind::ExpectedAddressSize(None)),
        }
    }

    /// Parse a top-level item (variable, function, entry, directive).
    fn parse_top_level_item(&mut self) -> Result<TopLevelItem, ParseError> {
        // Look for linkage specifiers first
        let linkage = self.parse_linkage()?;

        match self.peek()? {
            Some((_, Token::DottedIdent(DottedIdent::Simple(s)))) => {
                let s = s.clone();
                match s.as_bytes() {
                    b"entry" => {
                        self.next().unwrap();
                        Ok(TopLevelItem::Entry(self.parse_function(linkage)?))
                    }
                    b"func" => {
                        self.next().unwrap();
                        Ok(TopLevelItem::Function(self.parse_function(linkage)?))
                    }
                    b"file" => {
                        self.next().unwrap();
                        Ok(TopLevelItem::File(self.parse_file_directive()?))
                    }
                    // State spaces indicate variable declarations
                    b"reg" | b"sreg" | b"const" | b"global" | b"local" | b"param" | b"shared"
                    | b"tex" => Ok(TopLevelItem::Variable(self.parse_var_decl(linkage)?)),
                    // Generic directive
                    _ => Ok(TopLevelItem::Directive(self.parse_directive()?)),
                }
            }
            Some((span, tok)) => err_at(span, ParseErrorKind::UnexpectedToken(tok.clone())),
            None => err(ParseErrorKind::UnexpectedEof),
        }
    }

    /// Parse optional linkage specifier (.extern, .visible, .weak)
    fn parse_linkage(&mut self) -> Result<Linkage, ParseError> {
        match self.peek()? {
            Some((_, Token::DottedIdent(DottedIdent::Simple(s)))) => {
                if let Some(linkage) = Linkage::from_ascii(s) {
                    self.next().unwrap();
                    Ok(linkage)
                } else {
                    Ok(Linkage::None)
                }
            }
            _ => Ok(Linkage::None),
        }
    }

    // =========================================================================
    // Variable declarations
    // =========================================================================

    /// Parse a variable declaration.
    /// For top-level declarations that don't support comma-separated names.
    fn parse_var_decl(&mut self, linkage: Linkage) -> Result<VarDecl, ParseError> {
        // Get start span
        let start = self.peek()?.map(|(s, _)| s.0).unwrap_or(0);

        // Parse state space
        let (space, space_qualifier) = self.parse_state_space()?;

        // Parse optional alignment
        let align = self.parse_align()?;

        // Parse type
        let ty = self.parse_type()?;

        // Parse name (may be parameterized like `%r<100>`)
        let (name, param_count) = self.parse_var_name()?;

        // Parse optional array dimensions
        let array_dims = self.parse_array_dims()?;

        // Parse optional initializer
        let init = self.parse_initializer()?;

        // Get end span from semicolon
        let end = self.peek()?.map(|(s, _)| s.1).unwrap_or(start);

        // Consume semicolon
        self.expect(Token::Semicolon)?;

        Ok(VarDecl {
            span: Span(start, end),
            linkage,
            space,
            space_qualifier,
            align,
            ty,
            name,
            param_count,
            array_dims,
            init,
        })
    }

    /// Parse variable declarations that may have comma-separated names.
    /// e.g., `.reg .s32 a, b, c;` expands to three VarDecls sharing the same type.
    fn parse_var_decls(&mut self, linkage: Linkage) -> Result<Vec<VarDecl>, ParseError> {
        // Get start span before parsing state space
        let start = self.peek()?.map(|(s, _)| s.0).unwrap_or(0);

        // Parse state space
        let (space, space_qualifier) = self.parse_state_space()?;

        // Parse optional alignment
        let align = self.parse_align()?;

        // Parse type
        let ty = self.parse_type()?;

        let mut decls = Vec::new();

        loop {
            // Parse name (may be parameterized like `%r<100>`)
            let (name, param_count) = self.parse_var_name()?;

            // Parse optional array dimensions
            let array_dims = self.parse_array_dims()?;

            // Parse optional initializer (only for last name, or not at all for comma-separated)
            let init = if self.peek_tok()? == Some(&Token::Equals) {
                self.parse_initializer()?
            } else {
                None
            };

            // Check for comma (more names) or semicolon (end)
            let (end, done) = match self.peek()? {
                Some((span, Token::Comma)) => {
                    let end = span.1;
                    self.next().unwrap(); // consume comma, continue parsing names
                    (end, false)
                }
                Some((span, Token::Semicolon)) => {
                    let end = span.1;
                    self.next().unwrap(); // consume semicolon, done
                    (end, true)
                }
                Some((span, tok)) => {
                    return err_at(
                        span,
                        ParseErrorKind::ExpectedToken {
                            expected: Token::Semicolon,
                            found: Some(tok.clone()),
                        },
                    );
                }
                None => return err(ParseErrorKind::UnexpectedEof),
            };

            decls.push(VarDecl {
                span: Span(start, end),
                linkage,
                space,
                space_qualifier,
                align,
                ty,
                name,
                param_count,
                array_dims,
                init,
            });

            if done {
                break;
            }
        }

        Ok(decls)
    }

    /// Parse state space like `.global`, `.shared::cta`, etc.
    fn parse_state_space(
        &mut self,
    ) -> Result<(StateSpace, Option<StateSpaceQualifier>), ParseError> {
        match self.next()? {
            Some((span, Token::DottedIdent(DottedIdent::Simple(s)))) => {
                if let Some(space) = StateSpace::from_ascii(&s) {
                    Ok((space, None))
                } else {
                    err_at(span, ParseErrorKind::InvalidStateSpace(s))
                }
            }
            Some((span, Token::DottedIdent(DottedIdent::Qualified(parts)))) => {
                if parts.is_empty() {
                    return err(ParseErrorKind::EmptyQualifiedIdent);
                }
                let space = StateSpace::from_ascii(&parts[0]).ok_or_else(|| Locate {
                    path: None,
                    span: Some(span),
                    error: ParseErrorKind::InvalidStateSpace(parts[0].clone()),
                })?;
                let qualifier = if parts.len() > 1 {
                    match parts[1].as_bytes() {
                        b"cta" => Some(StateSpaceQualifier::Cta),
                        b"cluster" => Some(StateSpaceQualifier::Cluster),
                        b"entry" => Some(StateSpaceQualifier::Entry),
                        b"func" => Some(StateSpaceQualifier::Func),
                        _ => {
                            return err_at(
                                span,
                                ParseErrorKind::InvalidStateSpaceQualifier(parts[1].clone()),
                            );
                        }
                    }
                } else {
                    None
                };
                Ok((space, qualifier))
            }
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedStateSpace(Some(tok))),
            None => err(ParseErrorKind::ExpectedStateSpace(None)),
        }
    }

    /// Parse optional `.align N`. PTX requires the value to be a power of
    /// two, and the byte-layout code (`align_up` in `volta_analysis`)
    /// relies on it with only a debug assert - inert in release - so a
    /// zero or non-power-of-two value is rejected loudly here, at the one
    /// place every `.align` first lands, with the literal's span.
    /// (Values above `u32::MAX` are rejected by the same error rather
    /// than silently truncated by the cast.)
    fn parse_align(&mut self) -> Result<Option<u32>, ParseError> {
        if self.try_directive(ascii("align"))? {
            let span = self.peek()?.map(|(s, _)| s);
            let n = self.parse_int_literal()?;
            if !n.is_power_of_two() || n > u64::from(u32::MAX) {
                return Err(Locate {
                    path: None,
                    span,
                    error: ParseErrorKind::InvalidAlignment(n),
                });
            }
            Ok(Some(n as u32))
        } else {
            Ok(None)
        }
    }

    /// Parse a type like `.v4.f32`, `.u64`, etc.
    fn parse_type(&mut self) -> Result<Type, ParseError> {
        match self.next()? {
            Some((span, Token::DottedIdent(DottedIdent::Simple(s)))) => {
                // Could be just `.f32` or a vector width `.v4`
                if let Some(vec) = VecWidth::from_ascii(&s) {
                    // Need to parse the scalar type next
                    let scalar = self.parse_scalar_type()?;
                    Ok(Type {
                        vec: Some(vec),
                        scalar,
                    })
                } else if let Some(scalar) = ScalarType::from_ascii(&s) {
                    Ok(Type { vec: None, scalar })
                } else {
                    err_at(span, ParseErrorKind::InvalidType(s))
                }
            }
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedType(Some(tok))),
            None => err(ParseErrorKind::ExpectedType(None)),
        }
    }

    /// Parse the optional kernel-parameter pointer attribute.
    ///
    /// According to the PTX ISA (5.1.6.3), attributes are expected to be of the shape:
    /// `.ptr [.space] [.align N]`
    fn parse_ptr_attribute(&mut self) -> Result<Option<Ptr>, ParseError> {
        if !self.try_directive(ascii("ptr"))? {
            return Ok(None);
        }

        let space = match self.peek()? {
            Some((span, Token::DottedIdent(DottedIdent::Simple(s)))) => {
                if let Some(space) = PtrSpace::from_ascii(s) {
                    self.next().unwrap();
                    Some(space)
                } else if s.as_slice() == ascii("align") {
                    None
                } else {
                    return err_at(span, ParseErrorKind::InvalidPtrSpace(s.clone()));
                }
            }
            _ => None,
        };

        let align = self.parse_align()?;

        Ok(Some(Ptr { space, align }))
    }

    /// Parse a scalar type like `.f32`, `.s64`, etc.
    fn parse_scalar_type(&mut self) -> Result<ScalarType, ParseError> {
        match self.next()? {
            Some((span, Token::DottedIdent(DottedIdent::Simple(s)))) => ScalarType::from_ascii(&s)
                .ok_or(Locate {
                    path: None,
                    span: Some(span),
                    error: ParseErrorKind::InvalidType(s),
                }),
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedScalarType(Some(tok))),
            None => err(ParseErrorKind::ExpectedScalarType(None)),
        }
    }

    /// Parse variable name, possibly parameterized like `%r<100>`.
    fn parse_var_name(&mut self) -> Result<(AsciiString, Option<u32>), ParseError> {
        let name = self.expect_ident()?;

        // Check for parameterized syntax: `%r<100>`
        if self.peek_tok()? == Some(&Token::Less) {
            self.next().unwrap(); // consume <
            let count = self.parse_int_literal()? as u32;
            self.expect(Token::Greater)?;
            Ok((name.into_inner(), Some(count)))
        } else {
            Ok((name.into_inner(), None))
        }
    }

    /// Parse array dimensions like `[10]`, `[10][20]`, `[]` (unsized).
    fn parse_array_dims(&mut self) -> Result<Vec<Option<u32>>, ParseError> {
        let mut dims = Vec::new();

        while self.peek_tok()? == Some(&Token::LeftBracket) {
            self.next().unwrap(); // consume [

            // Check for unsized array `[]`
            if self.peek_tok()? == Some(&Token::RightBracket) {
                self.next().unwrap();
                dims.push(None);
            } else {
                let size = self.parse_int_literal()? as u32;
                self.expect(Token::RightBracket)?;
                dims.push(Some(size));
            }
        }

        Ok(dims)
    }

    /// Parse optional initializer `= value` or `= { ... }`.
    fn parse_initializer(&mut self) -> Result<Option<Initializer>, ParseError> {
        if self.peek_tok()? != Some(&Token::Equals) {
            return Ok(None);
        }
        self.next().unwrap(); // consume =

        Ok(Some(self.parse_init_value()?))
    }

    /// Parse an initializer value (scalar or aggregate).
    fn parse_init_value(&mut self) -> Result<Initializer, ParseError> {
        if self.peek_tok()? == Some(&Token::LeftBrace) {
            self.next().unwrap(); // consume {
            let mut values = Vec::new();

            if self.peek_tok()? != Some(&Token::RightBrace) {
                values.push(self.parse_init_value()?);
                while self.peek_tok()? == Some(&Token::Comma) {
                    self.next().unwrap();
                    values.push(self.parse_init_value()?);
                }
            }

            self.expect(Token::RightBrace)?;
            Ok(Initializer::Aggregate(values))
        } else {
            // Scalar value
            match self.next()? {
                Some((_, Token::SIntLit(n))) => Ok(Initializer::Scalar(InitValue::Int(n.value()))),
                Some((_, Token::UIntLit(n))) => Ok(Initializer::Scalar(InitValue::UInt(n.value()))),
                Some((_, Token::FloatLit(f))) => {
                    Ok(Initializer::Scalar(InitValue::Float(f.value())))
                }
                Some((_, Token::Ident(s))) => {
                    // Check for symbol+offset or generic(symbol)
                    if s.raw() == ascii("generic") {
                        self.expect(Token::LeftParen)?;
                        let sym = self.expect_ident()?;
                        self.expect(Token::RightParen)?;
                        Ok(Initializer::Scalar(InitValue::Generic(sym.into_inner())))
                    } else if self.peek_tok()? == Some(&Token::Plus)
                        || self.peek_tok()? == Some(&Token::Minus)
                    {
                        let is_neg = self.peek_tok()? == Some(&Token::Minus);
                        self.next().unwrap();
                        let offset = self.parse_int_literal()? as i64;
                        let offset = if is_neg { -offset } else { offset };
                        Ok(Initializer::Scalar(InitValue::SymbolOffset(
                            s.into_inner(),
                            offset,
                        )))
                    } else {
                        Ok(Initializer::Scalar(InitValue::Symbol(s.into_inner())))
                    }
                }
                Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedInitializer(Some(tok))),
                None => err(ParseErrorKind::ExpectedInitializer(None)),
            }
        }
    }

    // =========================================================================
    // Functions
    // =========================================================================

    /// Parse a function or entry definition.
    fn parse_function(&mut self, linkage: Linkage) -> Result<Function, ParseError> {
        // Parse optional return parameters: `(.param .b32 retval)`
        let return_params = if self.peek_tok()? == Some(&Token::LeftParen) {
            self.parse_param_list()?
        } else {
            Vec::new()
        };

        // Parse function name
        let (name_span, name) = self.parse_ident_with_span()?;

        // Parse input parameters
        let params = self.parse_param_list()?;

        // Parse performance-tuning directives (.maxntid, .minnctapersm, etc.)
        let perf_directives = self.parse_perf_directives()?;

        // Parse body or just declaration
        let body = if self.peek_tok()? == Some(&Token::LeftBrace) {
            Some(self.parse_function_body()?)
        } else if self.peek_tok()? == Some(&Token::Semicolon) {
            self.next().unwrap();
            None
        } else {
            None
        };

        Ok(Function {
            linkage,
            return_params,
            name: name.into_inner(),
            name_span,
            params,
            perf_directives,
            body,
        })
    }

    /// Parse parameter list `(.param .u64 ptr, .param .u32 n)`.
    fn parse_param_list(&mut self) -> Result<Vec<Parameter>, ParseError> {
        self.expect(Token::LeftParen)?;

        let mut params = Vec::new();
        if self.peek_tok()? != Some(&Token::RightParen) {
            params.push(self.parse_parameter()?);
            while self.peek_tok()? == Some(&Token::Comma) {
                self.next().unwrap();
                params.push(self.parse_parameter()?);
            }
        }

        self.expect(Token::RightParen)?;
        Ok(params)
    }

    /// Parse a single parameter.
    fn parse_parameter(&mut self) -> Result<Parameter, ParseError> {
        // Get start span
        let start = self.peek()?.map(|(s, _)| s.0).unwrap_or(0);

        let (space, _) = self.parse_state_space()?;
        let align = self.parse_align()?;
        let ty = self.parse_type()?;
        let ptr = self.parse_ptr_attribute()?;
        let (name_span, name) = self.parse_ident_with_span()?;

        // Parse array dimensions for byte arrays
        let mut end = name_span.1;
        let mut array_dims = Vec::new();
        while self.peek_tok()? == Some(&Token::LeftBracket) {
            self.next().unwrap();
            let size = self.parse_int_literal()? as u32;
            if let Some((span, _)) = self.peek()? {
                end = span.1;
            }
            self.expect(Token::RightBracket)?;
            array_dims.push(size);
        }

        Ok(Parameter {
            span: Span(start, end),
            space,
            align,
            ty,
            ptr,
            name: name.into_inner(),
            array_dims,
        })
    }

    /// Parse performance-tuning directives that appear after function parameters.
    fn parse_perf_directives(&mut self) -> Result<PerformanceDirectives, ParseError> {
        let mut directives = PerformanceDirectives::default();
        while let Some((_, Token::DottedIdent(DottedIdent::Simple(s)))) = self.peek()? {
            let s = s.clone();
            match s.as_bytes() {
                b"maxnreg" => {
                    self.next().unwrap();
                    directives.max_nreg = Some(self.parse_int_literal()? as u32);
                }
                b"maxntid" => {
                    self.next().unwrap();
                    directives.max_ntid = Some(self.parse_ntid_args()?);
                }
                b"reqntid" => {
                    self.next().unwrap();
                    directives.req_ntid = Some(self.parse_ntid_args()?);
                }
                b"minnctapersm" => {
                    self.next().unwrap();
                    directives.min_ncta_per_sm = Some(self.parse_int_literal()? as u32);
                }
                b"maxnctapersm" => {
                    self.next().unwrap();
                    directives.max_ncta_per_sm = Some(self.parse_int_literal()? as u32);
                }
                b"noreturn" => {
                    self.next().unwrap();
                    directives.noreturn = true;
                }
                b"pragma" => {
                    self.next().unwrap();
                    if let Some((_, Token::String(s))) = self.next()? {
                        directives.pragmas.push(s);
                    }
                }
                _ => break,
            }
        }
        Ok(directives)
    }

    /// Parse comma-separated ntid args: nx [, ny [, nz]]
    fn parse_ntid_args(&mut self) -> Result<(u32, Option<u32>, Option<u32>), ParseError> {
        let nx = self.parse_int_literal()? as u32;
        let ny = if self.peek_tok()? == Some(&Token::Comma) {
            self.next().unwrap();
            Some(self.parse_int_literal()? as u32)
        } else {
            None
        };
        let nz = if ny.is_some() && self.peek_tok()? == Some(&Token::Comma) {
            self.next().unwrap();
            Some(self.parse_int_literal()? as u32)
        } else {
            None
        };
        Ok((nx, ny, nz))
    }

    /// Parse function body `{ ... }`.
    fn parse_function_body(&mut self) -> Result<FunctionBody, ParseError> {
        self.expect(Token::LeftBrace)?;

        let mut statements = Vec::new();
        while self.peek_tok()? != Some(&Token::RightBrace) {
            self.parse_statements_into(&mut statements)?;
        }

        self.expect(Token::RightBrace)?;
        Ok(FunctionBody { statements })
    }

    /// Parse one or more statements and append to the given vector.
    /// This handles comma-separated variable declarations which expand to multiple statements.
    fn parse_statements_into(&mut self, statements: &mut Vec<Statement>) -> Result<(), ParseError> {
        match self.peek()? {
            // Variable declaration - may be comma-separated
            Some((_, Token::DottedIdent(DottedIdent::Simple(s))))
                if StateSpace::from_ascii(s).is_some() =>
            {
                let decls = self.parse_var_decls(Linkage::None)?;
                for decl in decls {
                    statements.push(Statement::Variable(decl));
                }
                Ok(())
            }
            // All other statements
            _ => {
                statements.push(self.parse_statement()?);
                Ok(())
            }
        }
    }

    // =========================================================================
    // Statements
    // =========================================================================

    /// Parse a statement (label, variable decl, instruction, directive).
    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek()? {
            // Ident could be a label (NAME:) or an instruction mnemonic
            Some((_, Token::Ident(ident))) => {
                if ident.as_instr().is_some() {
                    // It's an instruction mnemonic
                    self.parse_instruction_stmt()
                } else {
                    // Could be a label - peek ahead to see if followed by colon
                    let (ident_span, ident) = match self.next()? {
                        Some((span, Token::Ident(s))) => (span, s.into_inner()),
                        _ => unreachable!(),
                    };

                    if self.peek_tok()? == Some(&Token::Colon) {
                        self.next().unwrap(); // consume colon
                        Ok(Statement::Label(Label {
                            span: ident_span,
                            name: ident,
                        }))
                    } else {
                        // Not a label and not an instruction - error
                        err_at(
                            ident_span,
                            ParseErrorKind::UnexpectedIdentAtStatement(ident),
                        )
                    }
                }
            }

            // Predicated instruction: `@p` or `@!p`
            Some((_, Token::At)) => self.parse_instruction_stmt(),

            // Variable declaration or directive
            Some((_, Token::DottedIdent(_))) => {
                // Check if it's a state space (variable decl) or directive
                if let Some((_, Token::DottedIdent(DottedIdent::Simple(s)))) = self.peek()?
                    && StateSpace::from_ascii(s).is_some()
                {
                    // Variable declarations - handled by parse_statements_into
                    // If we get here, we're being called from a context that expects a single statement
                    // (like nested blocks). Use parse_var_decls and return the first one.
                    // Note: This path should rarely be hit since we updated nested blocks below.
                    let decls = self.parse_var_decls(Linkage::None)?;
                    if decls.len() == 1 {
                        return Ok(Statement::Variable(decls.into_iter().next().unwrap()));
                    } else {
                        // Multiple declarations - wrap in a block
                        return Ok(Statement::Block(
                            decls.into_iter().map(Statement::Variable).collect(),
                        ));
                    }
                }
                // Otherwise it's a directive
                Ok(Statement::Directive(self.parse_directive()?))
            }

            // Nested block
            Some((_, Token::LeftBrace)) => {
                self.next().unwrap();
                let mut stmts = Vec::new();
                while self.peek_tok()? != Some(&Token::RightBrace) {
                    self.parse_statements_into(&mut stmts)?;
                }
                self.expect(Token::RightBrace)?;
                Ok(Statement::Block(stmts))
            }

            Some((span, tok)) => err_at(span, ParseErrorKind::UnexpectedToken(tok.clone())),
            None => err(ParseErrorKind::UnexpectedEof),
        }
    }

    /// Parse an instruction statement (with optional predicate).
    fn parse_instruction_stmt(&mut self) -> Result<Statement, ParseError> {
        // Get start span (from predicate or opcode)
        let start = self.peek()?.map(|(s, _)| s.0).unwrap_or(0);

        let predicate = self.parse_predicate()?;
        let op = self.parse_instruction_op()?;

        // Get end span from semicolon
        let end = self.peek()?.map(|(s, _)| s.1).unwrap_or(start);
        self.expect(Token::Semicolon)?;

        Ok(Statement::Instruction(Instruction {
            span: Span(start, end),
            predicate,
            op,
        }))
    }

    /// Parse optional predicate `@p` or `@!p`.
    fn parse_predicate(&mut self) -> Result<Option<Predicate>, ParseError> {
        if self.peek_tok()? != Some(&Token::At) {
            return Ok(None);
        }
        self.next().unwrap(); // consume @

        let negated = if self.peek_tok()? == Some(&Token::Bang) {
            self.next().unwrap();
            true
        } else {
            false
        };

        let reg = self.expect_ident()?;
        Ok(Some(Predicate {
            negated,
            reg: reg.into_inner(),
        }))
    }

    /// Parse the instruction opcode and operands.
    fn parse_instruction_op(&mut self) -> Result<InstructionOp, ParseError> {
        // Get the full instruction with modifiers (e.g., `ld.param.u64`)
        let (kind, modifiers) = self.parse_opcode()?;

        // `call` has its own operand grammar with parenthesized return/argument
        // lists, e.g. `call.uni (retval0), __symexpf, (param0);`
        if kind == InstrKind::Call {
            return self.parse_call_operands(modifiers);
        }

        // Parse operands
        let operands = self.parse_operands()?;

        // For now, use Unparsed variant - strongly-typed parsing can be added later
        Ok(InstructionOp::Unparsed {
            kind,
            modifiers,
            operands,
        })
    }

    /// Parse call operands: `[(ret0, ret1, ...) ,] target [, (arg0, arg1, ...)]`.
    ///
    /// Direct calls only; the indirect-call prototype suffix is not supported.
    fn parse_call_operands(
        &mut self,
        modifiers: Vec<DottedIdent>,
    ) -> Result<InstructionOp, ParseError> {
        let uniform = modifiers
            .iter()
            .any(|m| matches!(m, DottedIdent::Simple(s) if s.as_slice() == ascii("uni")));

        // Optional parenthesized return-value list.
        let mut return_operands = Vec::new();
        if self.peek_tok()? == Some(&Token::LeftParen) {
            self.next().unwrap();
            return_operands.push(self.parse_operand()?);
            while self.peek_tok()? == Some(&Token::Comma) {
                self.next().unwrap();
                return_operands.push(self.parse_operand()?);
            }
            self.expect(Token::RightParen)?;
            self.expect(Token::Comma)?;
        }

        // Callee.
        let target = self.parse_operand()?;

        // Optional parenthesized argument list.
        let mut arguments = Vec::new();
        if self.peek_tok()? == Some(&Token::Comma) {
            self.next().unwrap();
            self.expect(Token::LeftParen)?;
            if self.peek_tok()? != Some(&Token::RightParen) {
                arguments.push(self.parse_operand()?);
                while self.peek_tok()? == Some(&Token::Comma) {
                    self.next().unwrap();
                    arguments.push(self.parse_operand()?);
                }
            }
            self.expect(Token::RightParen)?;
        }

        Ok(InstructionOp::Parsed(ParsedInstruction::Call(CallInstr {
            uniform,
            return_operands,
            target,
            arguments,
        })))
    }

    /// Parse opcode like `ld.param.u64` -> (InstrKind::Ld, [Simple("param"), Simple("u64")])
    fn parse_opcode(&mut self) -> Result<(InstrKind, Vec<DottedIdent>), ParseError> {
        // The lexer emits instruction mnemonics as Ident tokens (e.g., `st.async.weak`).
        // We use Ident::as_instruction() to find the longest matching mnemonic and get
        // any suffix modifiers (e.g., `st.async` + `.weak`).
        match self.next()? {
            Some((span, Token::Ident(ident))) => {
                // Process as_instr() result and collect modifiers immediately to avoid
                // borrowing ident past the point where we might need to move it.
                let instr_result = ident
                    .as_instr()
                    .map(|(kind, suffix)| (kind, parse_suffix_modifiers(suffix)));

                match instr_result {
                    Some((kind, mut modifiers)) => {
                        // Then collect any following DottedIdent tokens (space-separated)
                        while let Some((_, Token::DottedIdent(di))) = self.peek()? {
                            modifiers.push(di.clone());
                            self.next().unwrap();
                        }
                        Ok((kind, modifiers))
                    }
                    None => err_at(
                        span,
                        ParseErrorKind::ExpectedInstruction(Some(Token::Ident(ident))),
                    ),
                }
            }
            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedInstruction(Some(tok))),
            None => err(ParseErrorKind::ExpectedInstruction(None)),
        }
    }

    /// Parse comma-separated operands until semicolon.
    fn parse_operands(&mut self) -> Result<Vec<Operand>, ParseError> {
        let mut operands = Vec::new();

        // Empty operand list (e.g., `ret;`)
        if self.peek_tok()? == Some(&Token::Semicolon) {
            return Ok(operands);
        }

        operands.push(self.parse_operand()?);
        while self.peek_tok()? == Some(&Token::Comma) {
            self.next().unwrap();
            operands.push(self.parse_operand()?);
        }

        Ok(operands)
    }

    /// Parse a single operand.
    fn parse_operand(&mut self) -> Result<Operand, ParseError> {
        match self.peek()? {
            // Underscore (sink)
            Some((_, Token::Ident(s))) if s.raw() == ascii("_") => {
                self.next().unwrap();
                Ok(Operand::Underscore)
            }

            // Register or symbol
            Some((_, Token::Ident(_))) => {
                let name = self.expect_ident()?;

                // Check for vector element selector: %v.x
                if let Some((_, Token::DottedIdent(DottedIdent::Simple(comp)))) = self.peek()?
                    && let Some(vc) = VectorComponent::from_ascii(comp)
                {
                    self.next().unwrap();
                    return Ok(Operand::VectorElement(name.into_inner(), vc));
                }

                // Check for predicate pair: p|q
                if self.peek_tok()? == Some(&Token::Pipe) {
                    self.next().unwrap();
                    let other = self.expect_ident()?;
                    return Ok(Operand::PredicatePair(
                        name.into_inner(),
                        other.into_inner(),
                    ));
                }

                // Check for array element: var[offset]
                if self.peek_tok()? == Some(&Token::LeftBracket) {
                    self.next().unwrap(); // consume '['
                    let offset = self.parse_const_expr()?;
                    self.expect(Token::RightBracket)?;
                    return Ok(Operand::Address(Address {
                        base: AddressBase::Symbol(name.into_inner()),
                        offset: Some(Box::new(offset)),
                    }));
                }

                // Simple identifier - could be register, symbol, or label
                // The lowering phase will resolve what it actually refers to
                Ok(Operand::Ident(name.into_inner()))
            }

            // Negated predicate: !p
            Some((_, Token::Bang)) => {
                self.next().unwrap();
                let name = self.expect_ident()?;
                Ok(Operand::PredicateOperand {
                    negated: true,
                    name: name.into_inner(),
                })
            }

            // Address: [base + offset]
            Some((_, Token::LeftBracket)) => {
                self.next().unwrap();
                let addr = self.parse_address()?;
                self.expect(Token::RightBracket)?;
                Ok(Operand::Address(addr))
            }

            // Vector operand: {%r0, %r1, ...}
            Some((_, Token::LeftBrace)) => {
                self.next().unwrap();
                let mut elements = Vec::new();
                if self.peek_tok()? != Some(&Token::RightBrace) {
                    elements.push(self.parse_operand()?);
                    while self.peek_tok()? == Some(&Token::Comma) {
                        self.next().unwrap();
                        elements.push(self.parse_operand()?);
                    }
                }
                self.expect(Token::RightBrace)?;
                Ok(Operand::Vector(elements))
            }

            // Integer immediate
            Some((_, Token::SIntLit(_))) => {
                if let Some((_, Token::SIntLit(n))) = self.next()? {
                    Ok(Operand::ImmInt(n.value()))
                } else {
                    unreachable!()
                }
            }
            Some((_, Token::UIntLit(_))) => {
                if let Some((_, Token::UIntLit(n))) = self.next()? {
                    Ok(Operand::ImmUInt(n.value()))
                } else {
                    unreachable!()
                }
            }

            // Float immediate
            Some((_, Token::FloatLit(_))) => {
                if let Some((_, Token::FloatLit(f))) = self.next()? {
                    Ok(Operand::ImmFloat(f.value()))
                } else {
                    unreachable!()
                }
            }

            // Negative immediate: -123
            Some((_, Token::Minus)) => {
                self.next().unwrap(); // consume minus
                match self.next()? {
                    Some((_, Token::SIntLit(n))) => Ok(Operand::ImmInt(-n.value())),
                    Some((_, Token::UIntLit(n))) => Ok(Operand::ImmInt(-(n.value() as i64))),
                    Some((_, Token::FloatLit(f))) => Ok(Operand::ImmFloat(-f.value())),
                    Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedOperand(Some(tok))),
                    None => err(ParseErrorKind::ExpectedOperand(None)),
                }
            }

            Some((span, tok)) => err_at(span, ParseErrorKind::ExpectedOperand(Some(tok.clone()))),
            None => err(ParseErrorKind::ExpectedOperand(None)),
        }
    }

    /// Parse address inside brackets: `base`, `base + offset`, `base - offset`.
    fn parse_address(&mut self) -> Result<Address, ParseError> {
        // Parse base
        let base = match self.peek()? {
            Some((_, Token::Ident(_))) => {
                let name = self.expect_ident()?;
                if name.raw().starts_with(ascii("%")) {
                    AddressBase::Register(name.into_inner())
                } else {
                    AddressBase::Symbol(name.into_inner())
                }
            }
            Some((_, Token::SIntLit(n))) => {
                let n = n.value();
                self.next().unwrap();
                AddressBase::Immediate(n)
            }
            Some((_, Token::UIntLit(n))) => {
                let n = n.value() as i64;
                self.next().unwrap();
                AddressBase::Immediate(n)
            }
            Some((span, tok)) => {
                return err_at(span, ParseErrorKind::ExpectedAddressBase(Some(tok.clone())));
            }
            None => {
                return err(ParseErrorKind::ExpectedAddressBase(None));
            }
        };

        // Check for offset
        let offset = match self.peek_tok()? {
            Some(Token::Plus) | Some(Token::Minus) => {
                // Parse offset expression
                Some(Box::new(self.parse_const_expr()?))
            }
            _ => None,
        };

        Ok(Address { base, offset })
    }

    // =========================================================================
    // Directives
    // =========================================================================

    /// Parse a .file directive.
    fn parse_file_directive(&mut self) -> Result<FileDirective, ParseError> {
        let file_num = self.parse_int_literal()? as u32;

        // Parse filename string
        let filename = match self.next()? {
            Some((_, Token::String(s))) => s,
            Some((span, tok)) => return err_at(span, ParseErrorKind::ExpectedFilename(Some(tok))),
            None => return err(ParseErrorKind::ExpectedFilename(None)),
        };

        // Optional size and timestamp
        let mut size = None;
        let mut timestamp = None;

        while self.peek_tok()? == Some(&Token::Comma) {
            self.next().unwrap();
            match self.next()? {
                Some((_, Token::SIntLit(n))) => {
                    if size.is_none() {
                        size = Some(n.value() as u64);
                    } else {
                        timestamp = Some(n.value() as u64);
                    }
                }
                Some((_, Token::UIntLit(n))) => {
                    if size.is_none() {
                        size = Some(n.value());
                    } else {
                        timestamp = Some(n.value());
                    }
                }
                _ => break,
            }
        }

        Ok(FileDirective {
            file_num,
            filename,
            size,
            timestamp,
        })
    }

    /// Parse a generic directive.
    fn parse_directive(&mut self) -> Result<Directive, ParseError> {
        let (start, name) = match self.next()? {
            Some((span, Token::DottedIdent(DottedIdent::Simple(s)))) => (span.0, s),
            Some((span, Token::DottedIdent(DottedIdent::Qualified(parts)))) => {
                // Join parts with "::"
                let mut result = AsciiString::new();
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        result.push_slice(ascii("::"));
                    }
                    result.push_slice(part);
                }
                (span.0, result)
            }
            Some((span, other)) => {
                return err_at(
                    span,
                    ParseErrorKind::InvalidDirective(other.to_ascii_string()),
                );
            }
            None => return err(ParseErrorKind::UnexpectedEof),
        };

        // Collect arguments until semicolon
        let mut arguments = Vec::new();
        let mut end = start;
        while let Some((span, tok)) = self.next()? {
            end = span.1;
            if tok == Token::Semicolon {
                break;
            }
            arguments.push(tok);
        }

        if self.peek_tok()? == Some(&Token::Semicolon) {
            if let Some((span, _)) = self.peek()? {
                end = span.1;
            }
            self.next().unwrap();
        }

        Ok(Directive {
            span: Span(start, end),
            name,
            arguments,
        })
    }

    // =========================================================================
    // Constant Expression Parsing
    // =========================================================================

    /// Try to parse a cast like (.s64) or (.u64).
    /// Called after consuming '('. Returns Some(UnaryOp) if it's a cast,
    /// None if it's a regular parenthesized expression.
    fn try_parse_cast(&mut self) -> Result<Option<UnaryOp>, ParseError> {
        match self.peek()? {
            Some((_, Token::DottedIdent(DottedIdent::Simple(s)))) => {
                let cast_op = match s.as_str() {
                    "s64" => Some(UnaryOp::CastS64),
                    "u64" => Some(UnaryOp::CastU64),
                    _ => None,
                };

                if let Some(op) = cast_op {
                    self.next().unwrap(); // consume the dotted ident
                    self.expect(Token::RightParen)?;
                    Ok(Some(op))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Core Pratt parsing function.
    fn expr_bp(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        // Parse atom or prefix operator
        let mut lhs = match self.next()? {
            Some((_, Token::SIntLit(n))) => Expr::IntLitS(n.value()),
            Some((_, Token::UIntLit(n))) => Expr::IntLitU(n.value()),
            Some((_, Token::Ident(s))) => Expr::Ident(s.into_inner()),
            Some((_, Token::LeftParen)) => {
                // Check for cast: (.s64) or (.u64)
                if let Some(cast_op) = self.try_parse_cast()? {
                    let rhs = self.expr_bp(BP_UNARY)?;
                    Expr::Unary(cast_op, Box::new(rhs))
                } else {
                    // Regular parenthesized expression
                    let expr = self.expr_bp(0)?;
                    self.expect(Token::RightParen)?;
                    expr
                }
            }
            Some((_, ref op)) if prefix_binding_power(op).is_some() => {
                let op = op.clone();
                let ((), r_bp) = prefix_binding_power(&op).unwrap();
                let rhs = self.expr_bp(r_bp)?;
                Expr::Unary(token_to_unary_op(&op), Box::new(rhs))
            }
            Some((span, tok)) => return err_at(span, ParseErrorKind::UnexpectedToken(tok)),
            None => return err(ParseErrorKind::UnexpectedEof),
        };

        loop {
            let op = match self.peek()? {
                None => break,
                Some((_, op)) => op.clone(),
            };

            if let Some((l_bp, r_bp)) = infix_binding_power(&op) {
                if l_bp < min_bp {
                    break;
                }
                self.next().unwrap(); // consume operator

                // Special handling for ternary
                if matches!(op, Token::Question) {
                    let then_expr = self.expr_bp(0)?;
                    self.expect(Token::Colon)?;
                    let else_expr = self.expr_bp(r_bp)?;
                    lhs = Expr::Ternary(Box::new(lhs), Box::new(then_expr), Box::new(else_expr));
                } else {
                    let rhs = self.expr_bp(r_bp)?;
                    lhs = Expr::Binary(Box::new(lhs), token_to_binary_op(&op), Box::new(rhs));
                }
                continue;
            }

            break;
        }

        Ok(lhs)
    }

    /// Parse a constant expression.
    pub fn parse_const_expr(&mut self) -> Result<Expr, ParseError> {
        self.expr_bp(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascii::AsAscii;

    /// The volta_bench paper-benchmark kernel tree (PTX + CUDA sources).
    const KERNELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../volta_bench/kernels");

    fn parse(src: &[u8]) -> Result<Expr, ParseError> {
        let ascii_src = src.as_ascii_slice().expect("test input must be ASCII");
        Parser::new(ascii_src).parse_const_expr()
    }

    // Helper to build expressions more concisely
    fn int(n: i64) -> Expr {
        Expr::IntLitS(n)
    }

    fn uint(n: u64) -> Expr {
        Expr::IntLitU(n)
    }

    fn ascii_string(s: &str) -> AsciiString {
        AsciiString::try_from(s.to_owned()).unwrap()
    }

    fn ident(s: &str) -> Expr {
        Expr::Ident(ascii_string(s))
    }

    fn unary(op: UnaryOp, e: Expr) -> Expr {
        Expr::Unary(op, Box::new(e))
    }

    fn binary(l: Expr, op: BinaryOp, r: Expr) -> Expr {
        Expr::Binary(Box::new(l), op, Box::new(r))
    }

    fn ternary(c: Expr, t: Expr, e: Expr) -> Expr {
        Expr::Ternary(Box::new(c), Box::new(t), Box::new(e))
    }

    #[test]
    fn test_literals() {
        assert_eq!(parse(b"42"), Ok(int(42)));
        assert_eq!(parse(b"0xFF"), Ok(int(255)));
        assert_eq!(parse(b"42U"), Ok(uint(42)));
    }

    #[test]
    fn test_identifiers() {
        assert_eq!(parse(b"foo"), Ok(ident("foo")));
        assert_eq!(parse(b"%r0"), Ok(ident("%r0")));
    }

    #[test]
    fn test_unary_operators() {
        assert_eq!(parse(b"-1"), Ok(unary(UnaryOp::Neg, int(1))));
        assert_eq!(parse(b"+1"), Ok(unary(UnaryOp::Pos, int(1))));
        assert_eq!(parse(b"!0"), Ok(unary(UnaryOp::Not, int(0))));
        assert_eq!(parse(b"~0"), Ok(unary(UnaryOp::BitNot, int(0))));
    }

    #[test]
    fn test_unary_chaining() {
        // --1 should parse as -(-1)
        assert_eq!(
            parse(b"--1"),
            Ok(unary(UnaryOp::Neg, unary(UnaryOp::Neg, int(1))))
        );
    }

    #[test]
    fn test_binary_arithmetic() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3) due to precedence
        assert_eq!(
            parse(b"1 + 2 * 3"),
            Ok(binary(
                int(1),
                BinaryOp::Add,
                binary(int(2), BinaryOp::Mul, int(3))
            ))
        );
    }

    #[test]
    fn test_left_associativity() {
        // 1 - 2 - 3 should parse as (1 - 2) - 3
        assert_eq!(
            parse(b"1 - 2 - 3"),
            Ok(binary(
                binary(int(1), BinaryOp::Sub, int(2)),
                BinaryOp::Sub,
                int(3)
            ))
        );
    }

    #[test]
    fn test_parentheses() {
        // (1 + 2) * 3
        assert_eq!(
            parse(b"(1 + 2) * 3"),
            Ok(binary(
                binary(int(1), BinaryOp::Add, int(2)),
                BinaryOp::Mul,
                int(3)
            ))
        );
    }

    #[test]
    fn test_ternary() {
        // a ? b : c
        assert_eq!(
            parse(b"a ? b : c"),
            Ok(ternary(ident("a"), ident("b"), ident("c")))
        );
    }

    #[test]
    fn test_ternary_right_associativity() {
        // a ? b : c ? d : e should parse as a ? b : (c ? d : e)
        assert_eq!(
            parse(b"a ? b : c ? d : e"),
            Ok(ternary(
                ident("a"),
                ident("b"),
                ternary(ident("c"), ident("d"), ident("e"))
            ))
        );
    }

    #[test]
    fn test_casts() {
        assert_eq!(parse(b"(.s64) x"), Ok(unary(UnaryOp::CastS64, ident("x"))));
        assert_eq!(parse(b"(.u64) 42"), Ok(unary(UnaryOp::CastU64, int(42))));
    }

    #[test]
    fn test_cast_precedence() {
        // (.s64) 1 + 2 should parse as ((.s64) 1) + 2
        assert_eq!(
            parse(b"(.s64) 1 + 2"),
            Ok(binary(
                unary(UnaryOp::CastS64, int(1)),
                BinaryOp::Add,
                int(2)
            ))
        );
    }

    #[test]
    fn test_bitwise_precedence() {
        // 1 | 2 & 3 should parse as 1 | (2 & 3)
        assert_eq!(
            parse(b"1 | 2 & 3"),
            Ok(binary(
                int(1),
                BinaryOp::BitOr,
                binary(int(2), BinaryOp::BitAnd, int(3))
            ))
        );
    }

    #[test]
    fn test_comparison_and_logical() {
        // a < b && c > d
        assert_eq!(
            parse(b"a < b && c > d"),
            Ok(binary(
                binary(ident("a"), BinaryOp::Lt, ident("b")),
                BinaryOp::And,
                binary(ident("c"), BinaryOp::Gt, ident("d"))
            ))
        );
    }

    #[test]
    fn test_complex_expression() {
        // a && b || c ? x : y
        // Should parse as: ((a && b) || c) ? x : y
        assert_eq!(
            parse(b"a && b || c ? x : y"),
            Ok(ternary(
                binary(
                    binary(ident("a"), BinaryOp::And, ident("b")),
                    BinaryOp::Or,
                    ident("c")
                ),
                ident("x"),
                ident("y")
            ))
        );
    }

    #[test]
    fn test_shifts() {
        assert_eq!(parse(b"1 << 2"), Ok(binary(int(1), BinaryOp::Shl, int(2))));
        assert_eq!(parse(b"8 >> 1"), Ok(binary(int(8), BinaryOp::Shr, int(1))));
    }

    #[test]
    fn test_shift_precedence() {
        // 1 + 2 << 3 should parse as (1 + 2) << 3
        assert_eq!(
            parse(b"1 + 2 << 3"),
            Ok(binary(
                binary(int(1), BinaryOp::Add, int(2)),
                BinaryOp::Shl,
                int(3)
            ))
        );
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_empty_input() {
        assert!(matches!(
            parse(b"").map_err(|e| e.error),
            Err(ParseErrorKind::UnexpectedEof)
        ));
    }

    #[test]
    fn test_unclosed_paren() {
        assert!(matches!(
            parse(b"(1 + 2").map_err(|e| e.error),
            Err(ParseErrorKind::ExpectedToken {
                expected: Token::RightParen,
                found: None
            })
        ));
    }

    #[test]
    fn test_missing_ternary_colon() {
        assert!(matches!(
            parse(b"a ? b c").map_err(|e| e.error),
            Err(ParseErrorKind::ExpectedToken {
                expected: Token::Colon,
                found: Some(Token::Ident(_))
            })
        ));
    }

    #[test]
    fn test_unexpected_token() {
        // Semicolon is not valid in constant expressions
        assert!(matches!(
            parse(b";").map_err(|e| e.error),
            Err(ParseErrorKind::UnexpectedToken(Token::Semicolon))
        ));
    }

    #[test]
    fn test_binary_missing_rhs() {
        assert!(matches!(
            parse(b"1 +").map_err(|e| e.error),
            Err(ParseErrorKind::UnexpectedEof)
        ));
    }

    #[test]
    fn test_unary_missing_operand() {
        assert!(matches!(
            parse(b"-").map_err(|e| e.error),
            Err(ParseErrorKind::UnexpectedEof)
        ));
    }

    // =========================================================================
    // Deeply Nested Expressions
    // =========================================================================

    // Note: Removed deep nesting tests (deeply_nested_parens, nested_unary, long_chain_same_precedence)
    // These are adequately covered by test_parentheses, test_unary_chaining, and test_left_associativity

    #[test]
    fn test_mixed_mul_div_rem() {
        // 24 / 4 * 2 % 3 should be left-associative
        // (((24 / 4) * 2) % 3)
        assert_eq!(
            parse(b"24 / 4 * 2 % 3"),
            Ok(binary(
                binary(
                    binary(int(24), BinaryOp::Div, int(4)),
                    BinaryOp::Mul,
                    int(2)
                ),
                BinaryOp::Rem,
                int(3)
            ))
        );
    }

    // =========================================================================
    // Complex Precedence Tests
    // =========================================================================

    #[test]
    fn test_full_precedence_chain() {
        // Test that all precedence levels work correctly together
        // a || b && c | d ^ e & f == g < h + i * j
        // Should parse with correct precedence hierarchy
        assert_eq!(
            parse(b"a || b && c | d ^ e & f == g < h + i * j"),
            Ok(binary(
                ident("a"),
                BinaryOp::Or,
                binary(
                    ident("b"),
                    BinaryOp::And,
                    binary(
                        ident("c"),
                        BinaryOp::BitOr,
                        binary(
                            ident("d"),
                            BinaryOp::BitXor,
                            binary(
                                ident("e"),
                                BinaryOp::BitAnd,
                                binary(
                                    ident("f"),
                                    BinaryOp::Eq,
                                    binary(
                                        ident("g"),
                                        BinaryOp::Lt,
                                        binary(
                                            ident("h"),
                                            BinaryOp::Add,
                                            binary(ident("i"), BinaryOp::Mul, ident("j"))
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ))
        );
    }

    #[test]
    fn test_xor_between_or_and() {
        // 1 | 2 ^ 3 & 4 should parse as 1 | (2 ^ (3 & 4))
        assert_eq!(
            parse(b"1 | 2 ^ 3 & 4"),
            Ok(binary(
                int(1),
                BinaryOp::BitOr,
                binary(
                    int(2),
                    BinaryOp::BitXor,
                    binary(int(3), BinaryOp::BitAnd, int(4))
                )
            ))
        );
    }

    #[test]
    fn test_equality_vs_comparison() {
        // a == b < c should parse as a == (b < c)
        assert_eq!(
            parse(b"a == b < c"),
            Ok(binary(
                ident("a"),
                BinaryOp::Eq,
                binary(ident("b"), BinaryOp::Lt, ident("c"))
            ))
        );
    }

    #[test]
    fn test_all_comparison_operators() {
        assert_eq!(
            parse(b"a < b"),
            Ok(binary(ident("a"), BinaryOp::Lt, ident("b")))
        );
        assert_eq!(
            parse(b"a <= b"),
            Ok(binary(ident("a"), BinaryOp::Le, ident("b")))
        );
        assert_eq!(
            parse(b"a > b"),
            Ok(binary(ident("a"), BinaryOp::Gt, ident("b")))
        );
        assert_eq!(
            parse(b"a >= b"),
            Ok(binary(ident("a"), BinaryOp::Ge, ident("b")))
        );
        assert_eq!(
            parse(b"a == b"),
            Ok(binary(ident("a"), BinaryOp::Eq, ident("b")))
        );
        assert_eq!(
            parse(b"a != b"),
            Ok(binary(ident("a"), BinaryOp::Ne, ident("b")))
        );
    }

    // =========================================================================
    // Ternary Edge Cases
    // =========================================================================

    #[test]
    fn test_ternary_nesting_variants() {
        // Ternary in then branch: a ? b ? c : d : e parses as a ? (b ? c : d) : e
        assert_eq!(
            parse(b"a ? b ? c : d : e"),
            Ok(ternary(
                ident("a"),
                ternary(ident("b"), ident("c"), ident("d")),
                ident("e")
            ))
        );

        // Ternary in condition needs explicit parens for left-nesting
        assert_eq!(
            parse(b"(a ? b : c) ? d : e"),
            Ok(ternary(
                ternary(ident("a"), ident("b"), ident("c")),
                ident("d"),
                ident("e")
            ))
        );

        // Complex expressions in branches
        assert_eq!(
            parse(b"a + b ? c * d : e - f"),
            Ok(ternary(
                binary(ident("a"), BinaryOp::Add, ident("b")),
                binary(ident("c"), BinaryOp::Mul, ident("d")),
                binary(ident("e"), BinaryOp::Sub, ident("f"))
            ))
        );
    }

    // =========================================================================
    // Cast Edge Cases
    // =========================================================================

    #[test]
    fn test_cast_combinations() {
        // Cast of parenthesized expression
        assert_eq!(
            parse(b"(.s64) (a + b)"),
            Ok(unary(
                UnaryOp::CastS64,
                binary(ident("a"), BinaryOp::Add, ident("b"))
            ))
        );

        // Nested casts
        assert_eq!(
            parse(b"(.u64) (.s64) x"),
            Ok(unary(UnaryOp::CastU64, unary(UnaryOp::CastS64, ident("x"))))
        );

        // Cast of unary: (.s64) -x
        assert_eq!(
            parse(b"(.s64) -x"),
            Ok(unary(UnaryOp::CastS64, unary(UnaryOp::Neg, ident("x"))))
        );

        // Unary of cast: -(.s64) x parses as -((.s64) x)
        assert_eq!(
            parse(b"-(.s64) x"),
            Ok(unary(UnaryOp::Neg, unary(UnaryOp::CastS64, ident("x"))))
        );
    }

    #[test]
    fn test_non_cast_dotted_ident_in_parens() {
        // (.foo) is not a valid cast, so (.foo) should fail or parse differently
        // Since .foo is a DottedIdent but not s64/u64, it should be treated
        // as a parenthesized expression, but .foo isn't a valid expression start
        assert!(parse(b"(.foo)").is_err());
    }

    // =========================================================================
    // Unary vs Binary Disambiguation
    // =========================================================================

    #[test]
    fn test_unary_after_binary() {
        // 1 + -2 should parse as 1 + (-2)
        assert_eq!(
            parse(b"1 + -2"),
            Ok(binary(int(1), BinaryOp::Add, unary(UnaryOp::Neg, int(2))))
        );
    }

    #[test]
    fn test_unary_after_open_paren() {
        // (-1)
        assert_eq!(parse(b"(-1)"), Ok(unary(UnaryOp::Neg, int(1))));
    }

    #[test]
    fn test_unary_in_ternary() {
        // a ? -b : ~c
        assert_eq!(
            parse(b"a ? -b : ~c"),
            Ok(ternary(
                ident("a"),
                unary(UnaryOp::Neg, ident("b")),
                unary(UnaryOp::BitNot, ident("c"))
            ))
        );
    }

    // =========================================================================
    // Numeric Literals Edge Cases
    // =========================================================================

    #[test]
    fn test_zero() {
        assert_eq!(parse(b"0"), Ok(int(0)));
    }

    #[test]
    fn test_hex_literals() {
        assert_eq!(parse(b"0x0"), Ok(int(0)));
        assert_eq!(parse(b"0xDEADBEEF"), Ok(int(0xDEADBEEF)));
        assert_eq!(parse(b"0Xabc"), Ok(int(0xABC)));
    }

    #[test]
    fn test_binary_literals() {
        assert_eq!(parse(b"0b101"), Ok(int(5)));
        assert_eq!(parse(b"0B1111"), Ok(int(15)));
    }

    #[test]
    fn test_octal_literals() {
        assert_eq!(parse(b"0777"), Ok(int(511)));
        assert_eq!(parse(b"010"), Ok(int(8)));
    }

    #[test]
    fn test_unsigned_literals() {
        assert_eq!(parse(b"0U"), Ok(uint(0)));
        assert_eq!(parse(b"0xFFU"), Ok(uint(255)));
    }

    // =========================================================================
    // Real-world PTX-like Expressions
    // =========================================================================

    #[test]
    fn test_address_calculation() {
        // base + offset * sizeof_elem
        assert_eq!(
            parse(b"base + offset * 4"),
            Ok(binary(
                ident("base"),
                BinaryOp::Add,
                binary(ident("offset"), BinaryOp::Mul, int(4))
            ))
        );
    }

    #[test]
    fn test_alignment_check() {
        // (ptr & 0xF) == 0
        assert_eq!(
            parse(b"(ptr & 0xF) == 0"),
            Ok(binary(
                binary(ident("ptr"), BinaryOp::BitAnd, int(15)),
                BinaryOp::Eq,
                int(0)
            ))
        );
    }

    #[test]
    fn test_bit_manipulation() {
        // (x >> 4) & 0xF
        assert_eq!(
            parse(b"(x >> 4) & 0xF"),
            Ok(binary(
                binary(ident("x"), BinaryOp::Shr, int(4)),
                BinaryOp::BitAnd,
                int(15)
            ))
        );
    }

    #[test]
    fn test_conditional_value() {
        // flag ? val1 : val2
        assert_eq!(
            parse(b"flag ? 1 : 0"),
            Ok(ternary(ident("flag"), int(1), int(0)))
        );
    }

    #[test]
    fn test_bounds_check() {
        // idx >= 0 && idx < size
        assert_eq!(
            parse(b"idx >= 0 && idx < size"),
            Ok(binary(
                binary(ident("idx"), BinaryOp::Ge, int(0)),
                BinaryOp::And,
                binary(ident("idx"), BinaryOp::Lt, ident("size"))
            ))
        );
    }

    // =========================================================================
    // Module-Level Parsing Tests
    // =========================================================================

    fn parse_module(src: &[u8]) -> Result<Module, ParseError> {
        let ascii_src = src.as_ascii_slice().expect("test input must be ASCII");
        Parser::new(ascii_src).parse_module()
    }

    #[test]
    fn test_parse_version() {
        let src = b".version 7.0\n.target sm_70";
        let module = parse_module(src).unwrap();
        assert_eq!(module.version.major, 7);
        assert_eq!(module.version.minor, 0);
    }

    #[test]
    fn test_parse_target() {
        let src = b".version 7.0\n.target sm_70";
        let module = parse_module(src).unwrap();
        assert_eq!(module.target.archs[0].as_str(), "sm_70");
        assert_eq!(module.target.archs.len(), 1);
    }

    #[test]
    fn test_parse_address_size() {
        let src = b".version 7.0\n.target sm_70\n.address_size 64";
        let module = parse_module(src).unwrap();
        assert_eq!(module.address_size, Some(AddressSize::Bits64));
    }

    #[test]
    fn test_align_accepts_power_of_two() {
        let src = b".version 7.0\n.target sm_80\n.shared .align 16 .b8 buf[32];";
        let module = parse_module(src).unwrap();
        let Some(TopLevelItem::Variable(var)) = module.items.first() else {
            panic!("expected a module-scope variable declaration");
        };
        assert_eq!(var.align, Some(16));
    }

    /// Zero and non-power-of-two `.align` values are rejected at parse
    /// time with a located error. The byte-layout code (`align_up`)
    /// guards the power-of-two requirement only with a `debug_assert`,
    /// inert in release builds, so the parser is the loud guard - for
    /// variable declarations and parameters alike.
    #[test]
    fn test_align_rejects_zero_and_non_power_of_two() {
        let cases: &[(&[u8], u64)] = &[
            (b".version 7.0\n.target sm_80\n.shared .align 0 .b8 buf[32];", 0),
            (b".version 7.0\n.target sm_80\n.shared .align 3 .b8 buf[32];", 3),
            // A parameter's `.align` goes through the same chokepoint.
            (
                b".version 7.0\n.target sm_80\n.visible .entry k(.param .align 12 .b8 p[16]) { ret; }",
                12,
            ),
        ];
        for (src, bad) in cases {
            let err = parse_module(src).unwrap_err();
            assert_eq!(err.error, ParseErrorKind::InvalidAlignment(*bad));
            assert!(err.span.is_some(), "alignment error must carry a span");
        }
    }

    // =========================================================================
    // Directive Separation Tests (based on ptxas behavior)
    // =========================================================================

    #[test]
    fn test_directives_whitespace_variants() {
        // Whitespace type (space, tab, newline) should not affect parsing
        let variants: &[&[u8]] = &[
            b".version 7.0 .target sm_80 .address_size 64",
            b".version 7.0\t.target sm_80\t.address_size 64",
            b".version 7.0\n.target sm_80\n.address_size 64",
            b".version 7.0 .target sm_80\n.address_size 64",
        ];
        for src in variants {
            let module = parse_module(src).unwrap();
            assert_eq!(module.version.major, 7);
            assert_eq!(module.target.archs[0].as_str(), "sm_80");
            assert_eq!(module.address_size, Some(AddressSize::Bits64));
        }

        // Also test with entry on same line
        let src = b".version 7.0 .target sm_80 .address_size 64 .visible .entry test() { ret; }";
        let module = parse_module(src).unwrap();
        assert_eq!(module.items.len(), 1);
    }

    #[test]
    fn test_no_address_size() {
        // address_size is optional in ptxas
        let src = b".version 7.0\n.target sm_80";
        let module = parse_module(src).unwrap();
        assert_eq!(module.address_size, None);
    }

    #[test]
    fn test_target_with_texmode() {
        // Target can have comma-separated options
        let src = b".version 7.0\n.target sm_80, texmode_unified";
        let module = parse_module(src).unwrap();
        assert_eq!(module.target.archs[0].as_str(), "sm_80");
        assert_eq!(module.target.texmode, Some(Texmode::Unified));
    }

    #[test]
    fn test_target_with_multiple_options() {
        let src = b".version 7.0\n.target sm_80, debug, map_f64_to_f32";
        let module = parse_module(src).unwrap();
        assert_eq!(module.target.archs[0].as_str(), "sm_80");
        assert!(module.target.debug);
        assert!(module.target.map_f64_to_f32);
    }

    #[test]
    fn test_target_options_any_order() {
        // Options can appear in any order
        let src = b".version 7.0\n.target texmode_independent, sm_80, debug";
        let module = parse_module(src).unwrap();
        assert_eq!(module.target.archs[0].as_str(), "sm_80");
        assert_eq!(module.target.texmode, Some(Texmode::Independent));
        assert!(module.target.debug);
    }

    #[test]
    fn test_target_conflicting_texmode() {
        let src = b".version 7.0\n.target sm_80, texmode_unified, texmode_independent";
        let result = parse_module(src);
        assert!(matches!(
            result.map_err(|e| e.error),
            Err(ParseErrorKind::ConflictingTexmode)
        ));
    }

    #[test]
    fn test_target_invalid_option() {
        let src = b".version 7.0\n.target sm_80, invalid_option";
        let result = parse_module(src);
        assert!(matches!(
            result.map_err(|e| e.error),
            Err(ParseErrorKind::InvalidTargetOption(_))
        ));
    }

    #[test]
    fn test_empty_module_just_directives() {
        // ptxas accepts a module with just directives and no functions
        let src = b".version 7.0\n.target sm_80\n.address_size 64";
        let module = parse_module(src).unwrap();
        assert!(module.items.is_empty());
    }

    #[test]
    fn test_parse_simple_entry() {
        let src = b".version 7.0
.target sm_70
.address_size 64

.visible .entry test_kernel()
{
    ret;
}
";
        let module = parse_module(src).unwrap();
        assert_eq!(module.items.len(), 1);
        match &module.items[0] {
            TopLevelItem::Entry(func) => {
                assert_eq!(func.name.as_str(), "test_kernel");
                assert!(func.params.is_empty());
                assert!(func.body.is_some());
            }
            _ => panic!("Expected Entry"),
        }
    }

    #[test]
    fn test_function_name_is_instruction_mnemonic() {
        // Function names that happen to be instruction mnemonics should parse correctly.
        // This was a bug where `bar` (the barrier instruction) was incorrectly lexed as
        // Token::Instr instead of being treated as an identifier in function name context.
        let src = b".version 7.0
.target sm_70
.address_size 64

.func bar ( .param .b32 N, .param .b32 buffer[32] ) { }
.func add ( .param .u32 x ) { ret; }
.func mov ( ) { ret; }
";
        let module = parse_module(src).unwrap();
        assert_eq!(module.items.len(), 3);

        // Check that function names are parsed correctly even when they match instruction mnemonics
        match &module.items[0] {
            TopLevelItem::Function(func) => {
                assert_eq!(func.name.as_str(), "bar");
                assert_eq!(func.params.len(), 2);
            }
            _ => panic!("Expected Function"),
        }
        match &module.items[1] {
            TopLevelItem::Function(func) => {
                assert_eq!(func.name.as_str(), "add");
            }
            _ => panic!("Expected Function"),
        }
        match &module.items[2] {
            TopLevelItem::Function(func) => {
                assert_eq!(func.name.as_str(), "mov");
            }
            _ => panic!("Expected Function"),
        }
    }

    #[test]
    fn test_parse_entry_with_params() {
        let src = b".version 7.0
.target sm_70
.address_size 64

.visible .entry add_kernel(
    .param .u64 in_a,
    .param .u64 .ptr .global .align 16 in_b,
    .param .u32 n
)
{
    ret;
}
";
        let module = parse_module(src).unwrap();
        assert_eq!(module.items.len(), 1);
        match &module.items[0] {
            TopLevelItem::Entry(func) => {
                assert_eq!(func.name.as_str(), "add_kernel");
                assert_eq!(func.params.len(), 3);
                assert_eq!(func.params[0].name.as_str(), "in_a");
                assert_eq!(func.params[0].ty.scalar, ScalarType::U64);
                assert_eq!(func.params[0].ptr, None);
                assert_eq!(func.params[1].name.as_str(), "in_b");
                assert_eq!(func.params[1].ty.scalar, ScalarType::U64);
                assert_eq!(
                    func.params[1].ptr,
                    Some(Ptr {
                        space: Some(PtrSpace::Global),
                        align: Some(16),
                    })
                );
                assert_eq!(func.params[2].name.as_str(), "n");
                assert_eq!(func.params[2].ty.scalar, ScalarType::U32);
            }
            _ => panic!("Expected Entry"),
        }
    }

    #[test]
    fn test_parse_var_decl_parameterized() {
        let src = b".version 7.0
.target sm_70
.address_size 64

.visible .entry test()
{
    .reg .b32 %r<10>;
    ret;
}
";
        let module = parse_module(src).unwrap();
        match &module.items[0] {
            TopLevelItem::Entry(func) => {
                let body = func.body.as_ref().unwrap();
                match &body.statements[0] {
                    Statement::Variable(var) => {
                        assert_eq!(var.name.as_str(), "%r");
                        assert_eq!(var.param_count, Some(10));
                        assert_eq!(var.ty.scalar, ScalarType::B32);
                    }
                    _ => panic!("Expected Variable"),
                }
            }
            _ => panic!("Expected Entry"),
        }
    }

    #[test]
    fn test_parse_instruction_with_modifiers() {
        let src = b".version 7.0
.target sm_70
.address_size 64

.visible .entry test()
{
    .reg .u32 %r<2>;
    mov.u32 %r0, 42;
    ret;
}
";
        let module = parse_module(src).unwrap();
        match &module.items[0] {
            TopLevelItem::Entry(func) => {
                let body = func.body.as_ref().unwrap();
                // Statement 0: variable declaration
                // Statement 1: mov instruction
                match &body.statements[1] {
                    Statement::Instruction(instr) => {
                        assert!(instr.predicate.is_none());
                        match &instr.op {
                            InstructionOp::Unparsed {
                                kind, modifiers, ..
                            } => {
                                assert_eq!(*kind, InstrKind::Mov);
                                assert_eq!(
                                    modifiers,
                                    &vec![DottedIdent::Simple(ascii_string("u32"))]
                                );
                            }
                            _ => panic!("Expected Unparsed"),
                        }
                    }
                    _ => panic!("Expected Instruction"),
                }
            }
            _ => panic!("Expected Entry"),
        }
    }

    #[test]
    fn test_parse_predicated_instruction() {
        let src = b".version 7.0
.target sm_70
.address_size 64

.visible .entry test()
{
    .reg .pred %p<2>;
    @%p1 bra DONE;
DONE:
    ret;
}
";
        let module = parse_module(src).unwrap();
        match &module.items[0] {
            TopLevelItem::Entry(func) => {
                let body = func.body.as_ref().unwrap();
                // Statement 0: variable declaration
                // Statement 1: predicated branch
                match &body.statements[1] {
                    Statement::Instruction(instr) => {
                        let pred = instr.predicate.as_ref().unwrap();
                        assert!(!pred.negated);
                        assert_eq!(pred.reg.as_str(), "%p1");
                    }
                    _ => panic!("Expected Instruction"),
                }
                // Statement 2: label
                match &body.statements[2] {
                    Statement::Label(label) => {
                        assert_eq!(label.name.as_str(), "DONE");
                    }
                    _ => panic!("Expected Label"),
                }
            }
            _ => panic!("Expected Entry"),
        }
    }

    #[test]
    fn test_parse_memory_address() {
        let src = b".version 7.0
.target sm_70
.address_size 64

.visible .entry test()
{
    .reg .u64 %rd<2>;
    .reg .f32 %f<2>;
    ld.global.f32 %f1, [%rd1];
    ret;
}
";
        let module = parse_module(src).unwrap();
        match &module.items[0] {
            TopLevelItem::Entry(func) => {
                let body = func.body.as_ref().unwrap();
                // Statement 2: ld instruction
                match &body.statements[2] {
                    Statement::Instruction(instr) => match &instr.op {
                        InstructionOp::Unparsed { kind, operands, .. } => {
                            assert_eq!(*kind, InstrKind::Ld);
                            assert_eq!(operands.len(), 2);
                            // Second operand should be an address
                            match &operands[1] {
                                Operand::Address(addr) => match &addr.base {
                                    AddressBase::Register(r) => assert_eq!(r.as_str(), "%rd1"),
                                    _ => panic!("Expected Register base"),
                                },
                                _ => panic!("Expected Address operand"),
                            }
                        }
                        _ => panic!("Expected Unparsed"),
                    },
                    _ => panic!("Expected Instruction"),
                }
            }
            _ => panic!("Expected Entry"),
        }
    }

    #[test]
    fn test_parse_all_kernel_files() {
        // Test that all PTX files in volta_bench/kernels can be parsed
        for entry in std::fs::read_dir(KERNELS_DIR).expect("read kernels dir") {
            let dir = entry.expect("read kernels dir entry").path();
            if !dir.is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(&dir).expect("read category dir") {
                let path = entry.expect("read category dir entry").path();
                if path.extension().is_none_or(|e| e != "ptx") {
                    continue;
                }
                let src = std::fs::read(&path)
                    .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
                let result = parse_module(&src);
                assert!(
                    result.is_ok(),
                    "Failed to parse {}: {:?}",
                    path.display(),
                    result.err()
                );
            }
        }
    }

    // =========================================================================
    // PTX Examples Tests (from ptx_examples.txt)
    // =========================================================================

    /// Helper to wrap instruction snippets in a minimal module structure
    fn wrap_in_module(body: &str) -> String {
        format!(
            ".version 8.0\n.target sm_80\n.address_size 64\n\n.visible .entry test()\n{{\n{}\n    ret;\n}}\n",
            body
        )
    }

    /// Helper to parse a module and assert success
    fn parse_ok(src: &str) -> Module {
        let result = parse_module(src.as_bytes());
        assert!(
            result.is_ok(),
            "Failed to parse:\n{}\nError: {:?}",
            src,
            result.err()
        );
        result.unwrap()
    }

    #[test]
    fn test_parse_call_with_ret_and_args() {
        // The nvcc callseq idiom used for `__symexpf` in the paper benchmarks.
        let src = wrap_in_module(
            ".reg .f32 %f<3>;
    { // callseq 0, 0
    .reg .b32 temp_param_reg;
    .param .b32 param0;
    st.param.f32 [param0+0], %f1;
    .param .b32 retval0;
    call.uni (retval0),
    __symexpf,
    (
    param0
    );
    ld.param.f32 %f2, [retval0+0];
    } // callseq 0",
        );
        let module = parse_ok(&src);
        let func = match &module.items[0] {
            TopLevelItem::Entry(f) => f,
            _ => panic!("Expected Entry"),
        };
        let mut instrs = Vec::new();
        fn walk<'a>(stmts: &'a [Statement], out: &mut Vec<&'a Instruction>) {
            for s in stmts {
                match s {
                    Statement::Instruction(i) => out.push(i),
                    Statement::Block(b) => walk(b, out),
                    _ => {}
                }
            }
        }
        walk(&func.body.as_ref().unwrap().statements, &mut instrs);
        let call = instrs
            .iter()
            .find_map(|i| match &i.op {
                InstructionOp::Parsed(ParsedInstruction::Call(c)) => Some(c),
                _ => None,
            })
            .expect("call instruction should be parsed");
        assert!(call.uniform);
        assert_eq!(call.return_operands.len(), 1);
        assert_eq!(call.arguments.len(), 1);
        assert!(matches!(&call.target, Operand::Ident(name) if name.to_string() == "__symexpf"));
    }

    // --- Integer Arithmetic Examples ---
    // Note: Basic add/sub examples removed - covered by test_add_* and test_sub_* type tests

    #[test]
    fn test_example_5_mul_instructions() {
        // Example 5 (Page 143): mul instructions with modes
        let src = wrap_in_module(
            ".reg .s16 fxs, fys;
    .reg .s32 fa, x, y;
    .reg .s64 z;
    mul.wide.s16 fa,fxs,fys;
    mul.lo.s16 fa,fxs,fys;
    mul.wide.s32 z,x,y;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_6_mad_instructions() {
        // Example 6 (Page 144): mad instructions
        let src = wrap_in_module(
            ".reg .s32 a, b, c, d, r, p, q;
    .reg .pred pp;
    @pp mad.lo.s32 d,a,b,c;
    mad.lo.s32 r,p,q,r;",
        );
        parse_ok(&src);
    }

    // Note: min/max examples removed - covered by test_min_* and test_max_* type tests

    // --- Bit Manipulation Examples (Examples 16-24) ---

    #[test]
    fn test_example_16_popc() {
        // Example 16 (Page 156): popc instruction
        let src = wrap_in_module(
            ".reg .b32 d, a;
    .reg .b64 X;
    .reg .u32 cnt;
    popc.b32 d, a;
    popc.b64 cnt, X;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_18_bfind() {
        // Example 18 (Page 158): bfind instructions
        let src = wrap_in_module(
            ".reg .u32 d, a, cnt;
    .reg .s64 X;
    bfind.u32 d, a;
    bfind.shiftamt.s64 cnt, X;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_19_fns() {
        // Example 19 (Page 159): fns instruction
        let src = wrap_in_module(
            ".reg .b32 d;
    fns.b32 d, 0xaaaaaaaa, 3, 1;
    fns.b32 d, 0xaaaaaaaa, 3, -1;
    fns.b32 d, 0xaaaaaaaa, 2, 1;
    fns.b32 d, 0xaaaaaaaa, 2, -1;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_21_bfe() {
        // Example 21 (Page 162): bfe instruction
        let src = wrap_in_module(
            ".reg .b32 d, a, start, len;
    bfe.u32 d,a,start,len;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_25_dp4a() {
        // Example 25 (Page 166): dp4a instructions
        let src = wrap_in_module(
            ".reg .u32 d0, a0, b0, c0;
    .reg .s32 d1, a1, b1, c1;
    dp4a.u32.u32 d0, a0, b0, c0;
    dp4a.u32.s32 d1, a1, b1, c1;",
        );
        parse_ok(&src);
    }

    // --- Extended Precision Examples (Examples 27-32) ---

    #[test]
    fn test_example_27_extended_precision_add() {
        // Example 27 (Page 169): Extended precision addition
        let src = wrap_in_module(
            ".reg .u32 x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    .reg .pred p;
    @p add.cc.u32 x1,y1,z1;
    @p addc.cc.u32 x2,y2,z2;
    @p addc.cc.u32 x3,y3,z3;
    @p addc.u32 x4,y4,z4;",
        );
        parse_ok(&src);
    }

    // --- Floating-Point Examples (Examples 33-54) ---

    #[test]
    fn test_example_33_testp() {
        // Example 33 (Page 179): testp instructions
        let src = wrap_in_module(
            ".reg .f32 f0;
    .reg .f64 X;
    .reg .pred isnan, p;
    testp.notanumber.f32 isnan, f0;
    testp.infinite.f64 p, X;",
        );
        parse_ok(&src);
    }

    // Note: add float and fma examples removed - covered by test_add_* and test_fma_* type tests

    #[test]
    fn test_example_40_div() {
        // Example 40 (Page 191): div instructions
        let src = wrap_in_module(
            ".reg .f32 diam, circum, x, y, z;
    .reg .f64 xd, yd, zd;
    div.approx.ftz.f32 diam,circum,3.14159;
    div.full.ftz.f32 x, y, z;
    div.rn.f64 xd, yd, zd;",
        );
        parse_ok(&src);
    }

    // Note: min float example removed - covered by test_min_* type tests

    #[test]
    fn test_example_45_rcp() {
        // Example 45 (Page 201): rcp instructions
        let src = wrap_in_module(
            ".reg .f32 ri, r, xi, x;
    .reg .f64 xid, xd;
    rcp.approx.ftz.f32 ri,r;
    rcp.rn.ftz.f32 xi,x;
    rcp.rn.f64 xid,xd;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_47_sqrt() {
        // Example 47 (Page 204): sqrt instructions
        let src = wrap_in_module(
            ".reg .f32 r, x;
    .reg .f64 rd, xd;
    sqrt.approx.ftz.f32 r,x;
    sqrt.rn.ftz.f32 r,x;
    sqrt.rn.f64 rd,xd;",
        );
        parse_ok(&src);
    }

    // Note: Half-precision examples removed - covered by test_add_half_*, test_fma_half_*,
    // test_min_half_*, test_ex2_half_* type tests

    // Note: set/setp/slct examples removed - covered by test_set_* and test_slct_* type tests

    // --- Logic and Shift Examples (Examples 74-82) ---

    #[test]
    fn test_example_74_logic() {
        // Examples 74-77: and, or, xor, not
        let src = wrap_in_module(
            ".reg .b32 x, q, r, mask, fpvalue, sign, d;
    .reg .b16 dx, xi;
    .reg .pred p, pp, pq;
    and.b32 x,q,r;
    and.b32 sign,fpvalue,0x80000000;
    or.b32 mask,mask,0x00010001;
    or.pred p,pp,pq;
    xor.b32 d,q,r;
    xor.b16 dx,xi,0x0001;
    not.b32 mask,mask;
    not.pred p,pp;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_79_lop3() {
        // Example 79 (Page 257): lop3 instructions
        let src = wrap_in_module(
            ".reg .b32 d, a, b, c;
    .reg .pred p, q;
    lop3.b32 d, a, b, c, 0x40;
    lop3.or.b32 d|p, a, b, c, 0x3f, q;
    lop3.and.b32 d|p, a, b, c, 0x3f, q;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_80_shf() {
        // Example 80 (Page 259): shf instructions
        let src = wrap_in_module(
            ".reg .b32 r0, r1, r2, r3, r4, r5, r6, r7, n;
    shf.l.clamp.b32 r3,r1,r0,16;
    shf.l.clamp.b32 r7,r2,r3,n;
    shf.l.clamp.b32 r6,r1,r2,n;
    shf.l.clamp.b32 r5,r0,r1,n;
    shl.b32 r4,r0,n;
    shf.r.clamp.b32 r4,r0,r1,n;
    shf.r.clamp.b32 r5,r1,r2,n;
    shf.r.clamp.b32 r6,r2,r3,n;
    shr.s32 r7,r3,n;
    shf.r.clamp.b32 r1,r0,r0,n;
    shf.l.clamp.b32 r1,r0,r0,n;
    shf.r.clamp.b32 r0,r0,r1,n;",
        );
        parse_ok(&src);
    }

    // --- Data Movement Examples (Examples 83-91) ---

    #[test]
    fn test_example_83_mov() {
        // Example 83 (Page 266): mov instructions
        let src = wrap_in_module(
            ".reg .f32 d, a, k;
    .reg .u16 u, v;
    .reg .u32 ptr;
    mov.f32 d,a;
    mov.u16 u,v;
    mov.f32 k,0.1;",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_84_mov_vector() {
        // Example 84 (Page 268): mov with vector packing
        let src = wrap_in_module(
            ".reg .u16 a, b;
    .reg .b32 r1;
    .reg .b64 x;
    .reg .u32 lo, hi;
    .reg .b8 xb, yb, zb, wb;
    .reg .u8 rb, gb, bb, ab;
    .reg .b128 y;
    .reg .b64 b1, b2;
    mov.b32 r1,{a,b};
    mov.b64 {lo,hi}, x;
    mov.b32 r1,{xb,yb,zb,wb};
    mov.b32 {rb,gb,bb,ab},r1;
    mov.b64 {r1, r1}, x;
    mov.b128 {b1, b2}, y;
    mov.b128 y, {b1, b2};",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_87_prmt() {
        // Example 87 (Page 275): prmt instructions
        let src = wrap_in_module(
            ".reg .b32 r1, r2, r3, r4;
    prmt.b32 r1, r2, r3, r4;
    prmt.b32.f4e r1, r2, r3, r4;",
        );
        parse_ok(&src);
    }

    // --- Larger Example: Warp-level Scan (Example 85) ---

    #[test]
    fn test_example_85_warp_scan() {
        // Example 85 (Page 270): Warp-level inclusive plus scan
        let src = wrap_in_module(
            ".reg .b32 Rx, Ry;
    .reg .pred p;
    // Warp-level INCLUSIVE PLUS SCAN
    shfl.up.b32 Ry|p, Rx, 0x1, 0x0;
    @p add.f32 Rx, Ry, Rx;
    shfl.up.b32 Ry|p, Rx, 0x2, 0x0;
    @p add.f32 Rx, Ry, Rx;
    shfl.up.b32 Ry|p, Rx, 0x4, 0x0;
    @p add.f32 Rx, Ry, Rx;
    shfl.up.b32 Ry|p, Rx, 0x8, 0x0;
    @p add.f32 Rx, Ry, Rx;
    shfl.up.b32 Ry|p, Rx, 0x10, 0x0;
    @p add.f32 Rx, Ry, Rx;
    // BUTTERFLY REDUCTION
    shfl.bfly.b32 Ry, Rx, 0x10, 0x1f;
    add.f32 Rx, Ry, Rx;
    shfl.bfly.b32 Ry, Rx, 0x8, 0x1f;
    add.f32 Rx, Ry, Rx;
    shfl.bfly.b32 Ry, Rx, 0x4, 0x1f;
    add.f32 Rx, Ry, Rx;
    shfl.bfly.b32 Ry, Rx, 0x2, 0x1f;
    add.f32 Rx, Ry, Rx;
    shfl.bfly.b32 Ry, Rx, 0x1, 0x1f;
    add.f32 Rx, Ry, Rx;",
        );
        parse_ok(&src);
    }

    // --- Memory Examples (Examples 88-91) ---

    #[test]
    fn test_example_88_ld() {
        // Example 88 (Page 279): ld instructions
        let src = wrap_in_module(
            ".reg .f32 d;
    .reg .b32 Q0, Q1, Q2, Q3;
    .reg .s32 ds;
    .reg .b32 x;
    .reg .b64 xl;
    .reg .b16 r;
    .reg .u64 a, p;
    .shared .b32 sh[10];
    ld.global.f32 d,[a];
    ld.shared.v4.b32 {Q0, Q1, Q2, Q3},[sh];
    ld.const.s32 ds,[p+4];
    ld.local.b32 x,[p+-8];
    ld.local.b64 xl,[240];",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_example_91_st() {
        // Example 91 (Page 288): st instructions
        let src = wrap_in_module(
            ".reg .f32 b;
    .reg .s32 Q0, Q1, Q2, Q3, a;
    .reg .u64 addr, p, q;
    st.global.f32 [addr],b;
    st.local.b32 [q+4],a;
    st.global.v4.s32 [p],{Q0, Q1, Q2, Q3};",
        );
        parse_ok(&src);
    }

    #[test]
    fn test_array_element_address_syntax() {
        // PTX spec allows var[immOff] syntax for array element access
        // This is equivalent to [var+immOff]
        let src = wrap_in_module(
            ".reg .f32 f;
    .reg .s32 i;
    .shared .f32 arr[16];
    .global .s32 data[256];
    ld.shared.f32 f, arr[4];
    ld.shared.f32 f, arr[0];
    st.global.s32 data[8], i;
    ld.global.s32 i, data[12];",
        );
        parse_ok(&src);
    }

    // --- Synchronization Examples (Examples 117-121) ---

    #[test]
    fn test_example_barrier_and_fence() {
        // Various barrier and fence instructions
        let src = wrap_in_module(
            ".reg .u32 a;
    bar.sync a;
    bar.sync 0;
    membar.cta;
    membar.gl;
    membar.sys;
    fence.sc.cta;
    fence.acq_rel.gpu;",
        );
        parse_ok(&src);
    }

    // --- Atomic Examples ---

    #[test]
    fn test_atom_instructions() {
        // Atomic instructions
        let src = wrap_in_module(
            ".reg .u32 d, b;
    .reg .s32 ds, bs;
    .reg .u64 a;
    .shared .u32 sh;
    atom.global.add.u32 d, [a], b;
    atom.shared.max.s32 ds, [sh], bs;
    atom.global.cas.b32 d, [a], b, d;
    atom.global.exch.b32 d, [a], b;",
        );
        parse_ok(&src);
    }

    // --- Vote Examples ---

    #[test]
    fn test_vote_instructions() {
        // Vote instructions
        let src = wrap_in_module(
            ".reg .pred d, a;
    .reg .b32 db;
    vote.all.pred d, a;
    vote.any.pred d, a;
    vote.uni.pred d, a;
    vote.ballot.b32 db, a;",
        );
        parse_ok(&src);
    }

    // --- CVT Examples ---

    #[test]
    fn test_cvt_instructions() {
        // CVT type conversion instructions
        let src = wrap_in_module(
            ".reg .f32 f;
    .reg .f64 fd;
    .reg .s32 i;
    .reg .u16 u;
    .reg .f16 h;
    cvt.f32.s32 f, i;
    cvt.s32.f32 i, f;
    cvt.rni.s32.f32 i, f;
    cvt.f64.f32 fd, f;
    cvt.rn.f16.f32 h, f;
    cvt.f32.f16 f, h;",
        );
        parse_ok(&src);
    }

    // --- Control Flow Examples ---

    #[test]
    fn test_control_flow() {
        // Control flow instructions
        let src = wrap_in_module(
            ".reg .pred p;
    .reg .u32 x;
    setp.eq.u32 p, x, 0;
    @p bra DONE;
    add.u32 x, x, 1;
DONE:
    ret;",
        );
        // Note: we remove the explicit ret since wrap_in_module adds one
        let src = src.replace("    ret;\n    ret;", "    ret;");
        parse_ok(&src);
    }

    // =========================================================================
    // Strongly-Typed Instruction Tests
    //
    // These tests verify that instructions are parsed into the correct
    // strongly-typed enum variants with the expected field values.
    // =========================================================================

    use crate::instr_parse::parse_instruction;

    /// Helper to get the first instruction from a parsed module
    fn get_first_instruction(module: &Module) -> &Instruction {
        match &module.items[0] {
            TopLevelItem::Entry(func) => {
                let body = func.body.as_ref().expect("entry should have body");
                for stmt in &body.statements {
                    if let Statement::Instruction(instr) = stmt {
                        return instr;
                    }
                }
                panic!("No instruction found in module");
            }
            _ => panic!("Expected Entry"),
        }
    }

    /// Helper to parse an Unparsed instruction into a ParsedInstruction
    fn parse_instr(instr: &Instruction) -> ParsedInstruction {
        match &instr.op {
            InstructionOp::Unparsed {
                kind,
                modifiers,
                operands,
            } => parse_instruction(*kind, modifiers.clone(), operands.clone())
                .expect("instruction parsing should succeed"),
            InstructionOp::Parsed(p) => p.clone(),
        }
    }

    // --- AddInstr Strongly-Typed Tests ---

    #[test]
    fn test_add_integer_u32() {
        let src = wrap_in_module(
            ".reg .u32 %r<3>;
    add.u32 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Add(AddInstr::Integer { ty, .. }) => {
                assert_eq!(ty, ScalarType::U32);
            }
            _ => panic!("Expected AddInstr::Integer, got {:?}", parsed),
        }
    }

    #[test]
    fn test_add_integer_sat_s32() {
        let src = wrap_in_module(
            ".reg .s32 %r<3>;
    add.sat.s32 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Add(AddInstr::IntegerSat { sat, .. }) => {
                assert!(sat, "sat should be true");
            }
            _ => panic!("Expected AddInstr::IntegerSat, got {:?}", parsed),
        }
    }

    #[test]
    fn test_add_float32_with_modifiers() {
        let src = wrap_in_module(
            ".reg .f32 %f<3>;
    add.rz.ftz.sat.f32 %f0, %f1, %f2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Add(AddInstr::Float32 { rnd, ftz, sat, .. }) => {
                assert_eq!(rnd, Some(FpRound::Rz));
                assert!(ftz, "ftz should be true");
                assert!(sat, "sat should be true");
            }
            _ => panic!("Expected AddInstr::Float32, got {:?}", parsed),
        }
    }

    #[test]
    fn test_add_float64() {
        let src = wrap_in_module(
            ".reg .f64 %d<3>;
    add.rn.f64 %d0, %d1, %d2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Add(AddInstr::Float64 { rnd, .. }) => {
                assert_eq!(rnd, Some(FpRound::Rn));
            }
            _ => panic!("Expected AddInstr::Float64, got {:?}", parsed),
        }
    }

    #[test]
    fn test_add_half_f16() {
        let src = wrap_in_module(
            ".reg .f16 %h<3>;
    add.ftz.sat.f16 %h0, %h1, %h2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Add(AddInstr::HalfF16 { ftz, sat, ty, .. }) => {
                assert!(ftz, "ftz should be true");
                assert!(sat, "sat should be true");
                assert_eq!(ty, ScalarType::F16);
            }
            _ => panic!("Expected AddInstr::HalfF16, got {:?}", parsed),
        }
    }

    #[test]
    fn test_add_half_bf16() {
        let src = wrap_in_module(
            ".reg .bf16 %b<3>;
    add.rn.bf16 %b0, %b1, %b2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Add(AddInstr::HalfBf16 { rnd, ty, .. }) => {
                assert_eq!(rnd, Some(FpRound::Rn));
                assert_eq!(ty, ScalarType::Bf16);
            }
            _ => panic!("Expected AddInstr::HalfBf16, got {:?}", parsed),
        }
    }

    // --- SubInstr Strongly-Typed Tests ---

    #[test]
    fn test_sub_integer_s32() {
        // s32 can have sat modifier, so it uses IntegerSat variant even without sat
        let src = wrap_in_module(
            ".reg .s32 %r<3>;
    sub.s32 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Sub(SubInstr::IntegerSat { sat, .. }) => {
                assert!(!sat, "sat should be false for sub.s32 without .sat");
            }
            _ => panic!("Expected SubInstr::IntegerSat, got {:?}", parsed),
        }
    }

    #[test]
    fn test_sub_integer_u64() {
        // u64 cannot have sat modifier, so it uses Integer variant
        let src = wrap_in_module(
            ".reg .u64 %r<3>;
    sub.u64 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Sub(SubInstr::Integer { ty, .. }) => {
                assert_eq!(ty, ScalarType::U64);
            }
            _ => panic!("Expected SubInstr::Integer, got {:?}", parsed),
        }
    }

    #[test]
    fn test_sub_float32_ftz() {
        let src = wrap_in_module(
            ".reg .f32 %f<3>;
    sub.ftz.f32 %f0, %f1, %f2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Sub(SubInstr::Float32 { ftz, sat, .. }) => {
                assert!(ftz, "ftz should be true");
                assert!(!sat, "sat should be false");
            }
            _ => panic!("Expected SubInstr::Float32, got {:?}", parsed),
        }
    }

    // --- MinInstr Strongly-Typed Tests ---

    #[test]
    fn test_min_integer_s32() {
        // s32 can have relu modifier, so it uses IntegerRelu variant even without relu
        let src = wrap_in_module(
            ".reg .s32 %r<3>;
    min.s32 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Min(MinInstr::IntegerRelu { relu, ty, .. }) => {
                assert!(!relu, "relu should be false for min.s32 without .relu");
                assert_eq!(ty, ScalarType::S32);
            }
            _ => panic!("Expected MinInstr::IntegerRelu, got {:?}", parsed),
        }
    }

    #[test]
    fn test_min_integer_u32() {
        // u32 cannot have relu modifier, so it uses Integer variant
        let src = wrap_in_module(
            ".reg .u32 %r<3>;
    min.u32 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Min(MinInstr::Integer { ty, .. }) => {
                assert_eq!(ty, ScalarType::U32);
            }
            _ => panic!("Expected MinInstr::Integer, got {:?}", parsed),
        }
    }

    #[test]
    fn test_min_integer_relu() {
        let src = wrap_in_module(
            ".reg .s16x2 %r<3>;
    min.s16x2.relu %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Min(MinInstr::IntegerRelu { ty, .. }) => {
                assert_eq!(ty, ScalarType::S16x2);
            }
            _ => panic!("Expected MinInstr::IntegerRelu, got {:?}", parsed),
        }
    }

    #[test]
    fn test_min_float32_nan() {
        let src = wrap_in_module(
            ".reg .f32 %f<3>;
    min.NaN.f32 %f0, %f1, %f2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Min(MinInstr::Float32 { ftz, nan, .. }) => {
                assert!(!ftz, "ftz should be false");
                assert!(nan, "nan should be true");
            }
            _ => panic!("Expected MinInstr::Float32, got {:?}", parsed),
        }
    }

    #[test]
    fn test_min_half_f16_xorsign() {
        let src = wrap_in_module(
            ".reg .f16 %h<3>;
    min.xorsign.abs.f16 %h0, %h1, %h2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Min(MinInstr::HalfF16 {
                ftz,
                nan,
                xorsign_abs,
                ty,
                ..
            }) => {
                assert!(!ftz, "ftz should be false");
                assert!(!nan, "nan should be false");
                assert!(xorsign_abs, "xorsign_abs should be true");
                assert_eq!(ty, ScalarType::F16);
            }
            _ => panic!("Expected MinInstr::HalfF16, got {:?}", parsed),
        }
    }

    // --- MaxInstr Strongly-Typed Tests ---

    #[test]
    fn test_max_integer_u32() {
        let src = wrap_in_module(
            ".reg .u32 %r<3>;
    max.u32 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Max(MaxInstr::Integer { ty, .. }) => {
                assert_eq!(ty, ScalarType::U32);
            }
            _ => panic!("Expected MaxInstr::Integer, got {:?}", parsed),
        }
    }

    #[test]
    fn test_max_integer_relu() {
        let src = wrap_in_module(
            ".reg .s16x2 %r<3>;
    max.relu.s16x2 %r0, %r1, %r2;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Max(MaxInstr::IntegerRelu { ty, .. }) => {
                assert_eq!(ty, ScalarType::S16x2);
            }
            _ => panic!("Expected MaxInstr::IntegerRelu, got {:?}", parsed),
        }
    }

    // --- FmaInstr Strongly-Typed Tests ---

    #[test]
    fn test_fma_float32() {
        let src = wrap_in_module(
            ".reg .f32 %f<4>;
    fma.rn.ftz.f32 %f0, %f1, %f2, %f3;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Fma(FmaInstr::Float32 { rnd, ftz, sat, .. }) => {
                assert_eq!(rnd, FpRound::Rn);
                assert!(ftz, "ftz should be true");
                assert!(!sat, "sat should be false");
            }
            _ => panic!("Expected FmaInstr::Float32, got {:?}", parsed),
        }
    }

    #[test]
    fn test_fma_float64() {
        let src = wrap_in_module(
            ".reg .f64 %d<4>;
    fma.rn.f64 %d0, %d1, %d2, %d3;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Fma(FmaInstr::Float64 { rnd, .. }) => {
                assert_eq!(rnd, FpRound::Rn);
            }
            _ => panic!("Expected FmaInstr::Float64, got {:?}", parsed),
        }
    }

    #[test]
    fn test_fma_half_f16_relu() {
        let src = wrap_in_module(
            ".reg .f16 %h<4>;
    fma.rn.relu.f16 %h0, %h1, %h2, %h3;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Fma(FmaInstr::HalfF16Relu { rnd, ty, .. }) => {
                assert_eq!(rnd, FpRound::Rn);
                assert_eq!(ty, ScalarType::F16);
            }
            _ => panic!("Expected FmaInstr::HalfF16Relu, got {:?}", parsed),
        }
    }

    #[test]
    fn test_fma_half_f16_sat() {
        let src = wrap_in_module(
            ".reg .f16 %h<4>;
    fma.rn.sat.f16 %h0, %h1, %h2, %h3;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Fma(FmaInstr::HalfF16Sat { rnd, sat, ty, .. }) => {
                assert_eq!(rnd, FpRound::Rn);
                assert!(sat, "sat should be true");
                assert_eq!(ty, ScalarType::F16);
            }
            _ => panic!("Expected FmaInstr::HalfF16Sat, got {:?}", parsed),
        }
    }

    #[test]
    fn test_fma_oob() {
        let src = wrap_in_module(
            ".reg .f16 %h<4>;
    fma.rn.oob.f16 %h0, %h1, %h2, %h3;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Fma(FmaInstr::Oob { rnd, relu, ty, .. }) => {
                assert_eq!(rnd, FpRound::Rn);
                assert!(!relu, "relu should be false");
                assert_eq!(ty, ScalarType::F16);
            }
            _ => panic!("Expected FmaInstr::Oob, got {:?}", parsed),
        }
    }

    #[test]
    fn test_fma_oob_relu() {
        let src = wrap_in_module(
            ".reg .bf16 %b<4>;
    fma.rn.oob.relu.bf16 %b0, %b1, %b2, %b3;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Fma(FmaInstr::Oob { rnd, relu, ty, .. }) => {
                assert_eq!(rnd, FpRound::Rn);
                assert!(relu, "relu should be true");
                assert_eq!(ty, ScalarType::Bf16);
            }
            _ => panic!("Expected FmaInstr::Oob, got {:?}", parsed),
        }
    }

    // --- SetInstr Strongly-Typed Tests ---

    #[test]
    fn test_set_simple() {
        let src = wrap_in_module(
            ".reg .u32 %d;
    .reg .s32 %a, %b;
    set.lt.u32.s32 %d, %a, %b;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Set(SetInstr::Simple {
                cmp_op,
                dst_type,
                src_type,
                ..
            }) => {
                assert_eq!(cmp_op, CmpOp::Lt);
                assert_eq!(dst_type, ScalarType::U32);
                assert_eq!(src_type, ScalarType::S32);
            }
            _ => panic!("Expected SetInstr::Simple, got {:?}", parsed),
        }
    }

    #[test]
    fn test_set_with_bool_op() {
        let src = wrap_in_module(
            ".reg .f32 %d;
    .reg .s32 %a, %b, %c;
    set.lt.and.f32.s32 %d, %a, %b, %c;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Set(SetInstr::WithBoolOp {
                cmp_op,
                bool_op,
                dst_type,
                src_type,
                ..
            }) => {
                assert_eq!(cmp_op, CmpOp::Lt);
                assert_eq!(bool_op, BoolOp::And);
                assert_eq!(dst_type, ScalarType::F32);
                assert_eq!(src_type, ScalarType::S32);
            }
            _ => panic!("Expected SetInstr::WithBoolOp, got {:?}", parsed),
        }
    }

    // --- SetpInstr Strongly-Typed Tests ---

    #[test]
    fn test_setp_simple() {
        let src = wrap_in_module(
            ".reg .pred %p;
    .reg .u32 %a, %b;
    setp.eq.u32 %p, %a, %b;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Setp(SetpInstr::Simple { cmp_op, ty, .. }) => {
                assert_eq!(cmp_op, CmpOp::Eq);
                assert_eq!(ty, ScalarType::U32);
            }
            _ => panic!("Expected SetpInstr::Simple, got {:?}", parsed),
        }
    }

    #[test]
    fn test_setp_with_bool_op() {
        let src = wrap_in_module(
            ".reg .pred %p, %q;
    .reg .s32 %a, %b, %c;
    setp.lt.and.s32 %p|%q, %a, %b, %c;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Setp(SetpInstr::WithBoolOp {
                cmp_op,
                bool_op,
                ty,
                ..
            }) => {
                assert_eq!(cmp_op, CmpOp::Lt);
                assert_eq!(bool_op, BoolOp::And);
                assert_eq!(ty, ScalarType::S32);
            }
            _ => panic!("Expected SetpInstr::WithBoolOp, got {:?}", parsed),
        }
    }

    // --- SlctInstr Strongly-Typed Tests ---

    #[test]
    fn test_slct_integer() {
        let src = wrap_in_module(
            ".reg .u32 %d, %a, %b;
    .reg .s32 %c;
    slct.u32.s32 %d, %a, %b, %c;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Slct(SlctInstr::Integer { dst_type, .. }) => {
                assert_eq!(dst_type, ScalarType::U32);
            }
            _ => panic!("Expected SlctInstr::Integer, got {:?}", parsed),
        }
    }

    #[test]
    fn test_slct_float_with_ftz() {
        let src = wrap_in_module(
            ".reg .u64 %d, %a, %b;
    .reg .f32 %c;
    slct.ftz.u64.f32 %d, %a, %b, %c;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Slct(SlctInstr::Float { ftz, dst_type, .. }) => {
                assert!(ftz, "ftz should be true");
                assert_eq!(dst_type, ScalarType::U64);
            }
            _ => panic!("Expected SlctInstr::Float, got {:?}", parsed),
        }
    }

    // --- AbsInstr Strongly-Typed Tests ---

    #[test]
    fn test_abs_integer() {
        let src = wrap_in_module(
            ".reg .s32 %r<2>;
    abs.s32 %r0, %r1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Abs(AbsInstr::Integer { ty, .. }) => {
                assert_eq!(ty, ScalarType::S32);
            }
            _ => panic!("Expected AbsInstr::Integer, got {:?}", parsed),
        }
    }

    #[test]
    fn test_abs_float32_ftz() {
        let src = wrap_in_module(
            ".reg .f32 %f<2>;
    abs.ftz.f32 %f0, %f1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Abs(AbsInstr::Float32 { ftz, .. }) => {
                assert!(ftz, "ftz should be true");
            }
            _ => panic!("Expected AbsInstr::Float32, got {:?}", parsed),
        }
    }

    #[test]
    fn test_abs_float64() {
        let src = wrap_in_module(
            ".reg .f64 %d<2>;
    abs.f64 %d0, %d1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Abs(AbsInstr::Float64 { .. }) => {}
            _ => panic!("Expected AbsInstr::Float64, got {:?}", parsed),
        }
    }

    #[test]
    fn test_abs_half_f16() {
        let src = wrap_in_module(
            ".reg .f16 %h<2>;
    abs.ftz.f16 %h0, %h1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Abs(AbsInstr::HalfF16 { ftz, ty, .. }) => {
                assert!(ftz, "ftz should be true");
                assert_eq!(ty, ScalarType::F16);
            }
            _ => panic!("Expected AbsInstr::HalfF16, got {:?}", parsed),
        }
    }

    // --- NegInstr Strongly-Typed Tests ---

    #[test]
    fn test_neg_integer() {
        let src = wrap_in_module(
            ".reg .s32 %r<2>;
    neg.s32 %r0, %r1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Neg(NegInstr::Integer { ty, .. }) => {
                assert_eq!(ty, ScalarType::S32);
            }
            _ => panic!("Expected NegInstr::Integer, got {:?}", parsed),
        }
    }

    #[test]
    fn test_neg_float32() {
        let src = wrap_in_module(
            ".reg .f32 %f<2>;
    neg.ftz.f32 %f0, %f1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Neg(NegInstr::Float32 { ftz, .. }) => {
                assert!(ftz, "ftz should be true");
            }
            _ => panic!("Expected NegInstr::Float32, got {:?}", parsed),
        }
    }

    // --- Ex2Instr Strongly-Typed Tests ---

    #[test]
    fn test_ex2_float32() {
        let src = wrap_in_module(
            ".reg .f32 %f<2>;
    ex2.approx.ftz.f32 %f0, %f1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Ex2(Ex2Instr::Float32 { ftz, .. }) => {
                assert!(ftz, "ftz should be true");
            }
            _ => panic!("Expected Ex2Instr::Float32, got {:?}", parsed),
        }
    }

    #[test]
    fn test_ex2_half_f16() {
        let src = wrap_in_module(
            ".reg .f16 %h<2>;
    ex2.approx.f16 %h0, %h1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Ex2(Ex2Instr::HalfF16 { ty, .. }) => {
                assert_eq!(ty, ScalarType::F16);
            }
            _ => panic!("Expected Ex2Instr::HalfF16, got {:?}", parsed),
        }
    }

    #[test]
    fn test_ex2_half_bf16_with_ftz() {
        // bf16 requires ftz
        let src = wrap_in_module(
            ".reg .bf16 %b<2>;
    ex2.approx.ftz.bf16 %b0, %b1;",
        );
        let module = parse_ok(&src);
        let instr = get_first_instruction(&module);
        let parsed = parse_instr(instr);

        match parsed {
            ParsedInstruction::Ex2(Ex2Instr::HalfBf16 { ty, .. }) => {
                assert_eq!(ty, ScalarType::Bf16);
            }
            _ => panic!("Expected Ex2Instr::HalfBf16, got {:?}", parsed),
        }
    }
}
