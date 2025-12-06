
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

TOKEN_REGEX = re.compile(
    r"""\s*(
    (?P<AND>and)|
    (?P<OR>or)|
    (?P<NOT>not)|
    (?P<EQ>==)|
    (?P<NEQ>!=)|
    (?P<LPAREN>\()|
    (?P<RPAREN>\))|
    (?P<TRUE>true)|
    (?P<FALSE>false)|
    (?P<STRING>\"[^\"]*\")|
    (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    )""",
    re.VERBOSE,
)

@dataclass
class Token:
    kind: str
    value: str

class RuleParseError(Exception):
    pass

def tokenize(rule: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(rule):
        match = TOKEN_REGEX.match(rule, pos)
        if not match:
            raise RuleParseError(f"Unexpected token at position {pos} in rule: {rule!r}")
        kind = None
        value = match.group(0).strip()
        for name in (
            "AND",
            "OR",
            "NOT",
            "EQ",
            "NEQ",
            "LPAREN",
            "RPAREN",
            "TRUE",
            "FALSE",
            "STRING",
            "IDENT",
        ):
            if match.group(name):
                kind = name
                break
        if not kind:
            raise RuleParseError(f"Could not determine token kind for {value!r}")
        tokens.append(Token(kind=kind, value=value))
        pos = match.end()
    return tokens

# AST nodes

class Expr:
    pass

@dataclass
class Literal(Expr):
    value: Any

@dataclass
class Identifier(Expr):
    name: str

@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr

@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, kind: str) -> Token:
        tok = self.peek()
        if not tok or tok.kind != kind:
            raise RuleParseError(f"Expected {kind} but got {tok.kind if tok else 'EOF'}")
        self.pos += 1
        return tok

    def parse(self) -> Expr:
        expr = self.parse_or()
        if self.peek() is not None:
            raise RuleParseError("Unexpected trailing tokens in rule")
        return expr

    def parse_or(self) -> Expr:
        node = self.parse_and()
        while True:
            tok = self.peek()
            if tok and tok.kind == "OR":
                self.consume("OR")
                right = self.parse_and()
                node = BinaryOp(op="or", left=node, right=right)
            else:
                break
        return node

    def parse_and(self) -> Expr:
        node = self.parse_not()
        while True:
            tok = self.peek()
            if tok and tok.kind == "AND":
                self.consume("AND")
                right = self.parse_not()
                node = BinaryOp(op="and", left=node, right=right)
            else:
                break
        return node

    def parse_not(self) -> Expr:
        tok = self.peek()
        if tok and tok.kind == "NOT":
            self.consume("NOT")
            operand = self.parse_not()
            return UnaryOp(op="not", operand=operand)
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        tok = self.peek()
        if not tok:
            raise RuleParseError("Unexpected end of input")
        if tok.kind == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_or()
            self.consume("RPAREN")
            return expr
        if tok.kind == "TRUE":
            self.consume("TRUE")
            return Literal(True)
        if tok.kind == "FALSE":
            self.consume("FALSE")
            return Literal(False)
        if tok.kind == "STRING":
            self.consume("STRING")
            # Strip quotes
            return Literal(tok.value[1:-1])
        if tok.kind == "IDENT":
            # Could be bare identifier or comparison
            ident = self.consume("IDENT")
            next_tok = self.peek()
            if next_tok and next_tok.kind in ("EQ", "NEQ"):
                op = "==" if next_tok.kind == "EQ" else "!="
                self.consume(next_tok.kind)
                literal = self.parse_atom()
                if not isinstance(literal, Literal):
                    raise RuleParseError("Right side of comparison must be a literal")
                return BinaryOp(op=op, left=Identifier(ident.value), right=literal)
            else:
                # Bare identifier is treated as boolean flag
                return Identifier(ident.value)
        raise RuleParseError(f"Unexpected token {tok.kind} with value {tok.value!r}")

def parse_rule(rule: str) -> Expr:
    tokens = tokenize(rule)
    parser = Parser(tokens)
    return parser.parse()

def eval_expr(expr: Expr, features: Dict[str, Any]) -> bool:
    if isinstance(expr, Literal):
        return bool(expr.value)
    if isinstance(expr, Identifier):
        value = features.get(expr.name, False)
        return bool(value)
    if isinstance(expr, UnaryOp):
        if expr.op == "not":
            return not eval_expr(expr.operand, features)
        raise RuleParseError(f"Unknown unary operator {expr.op!r}")
    if isinstance(expr, BinaryOp):
        if expr.op in ("and", "or"):
            left = eval_expr(expr.left, features)
            right = eval_expr(expr.right, features)
            return (left and right) if expr.op == "and" else (left or right)
        if expr.op in ("==", "!="):
            if not isinstance(expr.left, Identifier):
                raise RuleParseError("Left side of comparison must be an identifier")
            left_val = features.get(expr.left.name, None)
            right_val = expr.right.value
            if expr.op == "==":
                return left_val == right_val
            return left_val != right_val
        raise RuleParseError(f"Unknown binary operator {expr.op!r}")
    raise RuleParseError(f"Unsupported expression node {type(expr).__name__}")

def evaluate_rule(rule: str, features: Dict[str, Any]) -> bool:
    """Evaluate a rule string against a features dict.

    Returns False if the rule cannot be parsed or evaluated. The caller can
    treat such cases as non matches and handle them separately.
    """
    try:
        expr = parse_rule(rule)
        return eval_expr(expr, features)
    except RuleParseError:
        return False
