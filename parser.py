"""
Parser for the simple C-like language.

Overview and approach:
- This parser implements a small, hand-written recursive/Pratt-style parser
    for statements and expressions. Expressions use a Pratt-like approach with
    a precedence table stored in `self.precedence`. This keeps expression parsing
    concise while correctly handling operator precedence and associativity.

Key points:
- Expression parsing:
    - `parse_primary()` recognizes literals, identifiers, parenthesized
        expressions and prefixes (unary operators). It also delegates to
        `parse_block()` when a block is encountered.
    - `parse_postfix()` handles postfix constructs such as function calls and
        array indexing (both have the highest precedence).
    - `parse_binary_expression()` implements the Pratt loop: while the next
        token has precedence >= the current minimum, bind that operator and parse
        the right-hand side using a higher minimum precedence. Assignment is
        treated specially as right-associative.

- Statement parsing:
    - `parse_statement()` recognizes declarations (`int`, `bool`, `void`),
        control flow (`if`, `while`), blocks, and `return`. It falls back to an
        expression statement for other constructs.

- Functions and prototypes:
    - A function declaration is recognized by lookahead when a type token is
        followed by an identifier and a left parenthesis: `type ident(`.
    - Prototypes (declarations without bodies) are allowed: a prototype is a
        function signature that ends with `;`. A definition has a `{ ... }` body.
    - When parsing a function definition, the parser creates a nested
        `SymbolTable` for the function body and declares parameter names/types
        in that scope so that references to parameters inside the body resolve.

Examples:
    - Prototype: `int add(int a, int b);`
    - Definition: `int add(int a, int b) { return a + b; }`

Notes:
- The parser depends on `SymbolTable` to check for undeclared identifiers
    when an identifier is parsed; therefore functions must be declared (as a
    prototype or earlier definition) before they are called.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from tokens import Token, TokenType
from ast_nodes import *
from symbols import SymbolTable, SymbolType


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else Token(TokenType.EOF, None)
        self.symbol_table = SymbolTable()

        # Operator precedence table (higher = tighter binding)
        self.precedence: Dict[TokenType, int] = {
            TokenType.ASSIGN: 0,
            TokenType.OR: 1,
            TokenType.AND: 2,
            TokenType.EQ: 3,
            TokenType.NEQ: 3,
            TokenType.LT: 4,
            TokenType.GT: 4,
            TokenType.LTE: 4,
            TokenType.GTE: 4,
            TokenType.PLUS: 5,
            TokenType.MINUS: 5,
            TokenType.STAR: 6,
            TokenType.SLASH: 6,
            TokenType.MOD: 6,
            TokenType.LBRACKET: 7,
            TokenType.LPAREN: 7,
            TokenType.NOT: 8,
        }

    def peek(self) -> Token:
        """Return next token without consuming it."""
        return (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else Token(TokenType.EOF, None)
        )

    def advance(self) -> Token:
        """Move to next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = Token(TokenType.EOF, None)
        return self.current

    def expect(self, expected_type: TokenType, message: Optional[str] = None) -> Token:
        """Expect and consume token of given type."""
        if self.current.type == expected_type:
            token = self.current
            self.advance()
            return token

        msg = message or f"Expected {expected_type}, got {self.current.type}"
        raise SyntaxError(msg)

    def match(self, token_type: TokenType) -> bool:
        """Check if current token matches type, consume if true."""
        if self.current.type == token_type:
            self.advance()
            return True
        return False

    def get_precedence(self, token_type: TokenType) -> int:
        """Get precedence for operator token type."""
        return self.precedence.get(token_type, 0)

    def parse_type(self) -> SymbolType:
        """Parse a type: int, bool, int[]."""
        match self.current.type:
            case TokenType.INT_TYPE:
                self.advance()
                # Check for array type: int[]
                if self.current.type == TokenType.LBRACKET:
                    self.advance()
                    self.expect(TokenType.RBRACKET)
                    return SymbolType.INT_ARRAY
                return SymbolType.INT
            case TokenType.BOOL_TYPE:
                self.advance()
                return SymbolType.BOOL
            case TokenType.VOID_TYPE:
                self.advance()
                return SymbolType.VOID
            case _:
                raise SyntaxError(f"Expected type, got {self.current.type}")

    def parse_primary(self) -> ASTNode:
        """Parse primary expressions (literals, identifiers, parenthesized)."""
        token = self.current

        match token.type:
            case TokenType.INTEGER:
                self.advance()
                return IntLiteralNode(value=token.value)

            case TokenType.TRUE:
                self.advance()
                return BoolLiteralNode(value=True)

            case TokenType.FALSE:
                self.advance()
                return BoolLiteralNode(value=False)

            case TokenType.IDENTIFIER:
                var_name = token.value
                if not self.symbol_table.exists(var_name):
                    raise SyntaxError(f"Undeclared variable '{var_name}'")

                symbol = self.symbol_table.lookup(var_name)
                self.advance()
                return IdentifierNode(
                    # `left` may be an IdentifierNode (or another expression
                    # evaluating to a function). We create a FunctionCallNode
                    # regardless; the type checker will validate the signature.
                    name=var_name,
                    symbol_type=symbol.type,
                    is_array=symbol.is_array,
                    is_function=symbol.is_function,
                    parameters=(
                        symbol.parameters if symbol.parameters is not None else []
                    ),
                )

            case TokenType.LPAREN:
                self.advance()  # Consume '('
                # Pratt parsing loop: repeatedly bind operators with sufficient
                # precedence to the current left-hand expression.
                expr = self.parse_expression()
                self.expect(TokenType.RPAREN)
                return expr

            case TokenType.MINUS:  # Unary minus
                self.advance()
                right = self.parse_primary()
                return UnaryOpNode(operator="-", right=right)

            case TokenType.NOT:  # Unary not
                self.advance()
                right = self.parse_primary()
                return UnaryOpNode(operator="!", right=right)

            case TokenType.LBRACE:
                return self.parse_block()

            case _:
                raise SyntaxError(f"Unexpected token: {token}")

    def parse_postfix(self, left: ASTNode) -> ASTNode:
        """Parse postfix expressions (array indexing, function calls)."""
        while True:
            match self.current.type:
                case TokenType.LBRACKET:
                    # Array indexing
                    if not isinstance(left, IdentifierNode) or not left.is_array:
                        raise SyntaxError("Cannot index non-array type")

                    self.advance()
                    index = self.parse_expression()
                    self.expect(TokenType.RBRACKET)
                    left = ArrayIndexNode(array=left, index=index)

                case TokenType.LPAREN:
                    # Function call
                    self.advance()
                    args: List[ASTNode] = []

                    if self.current.type != TokenType.RPAREN:
                        args.append(self.parse_expression())
                        while self.current.type == TokenType.COMMA:
                            self.advance()
                            args.append(self.parse_expression())

                    self.expect(TokenType.RPAREN)
                    # Propagate function signature info if available on identifier
                    left = FunctionCallNode(function=left, arguments=args)

                case _:
                    break

        return left

    def parse_binary_expression(
        self, left: ASTNode, min_precedence: int = 0
    ) -> ASTNode:
        """Parse binary expressions using Pratt parsing."""
        while True:
            token = self.current

            # Check if we should stop
            if token.type == TokenType.EOF or token.type in (
                TokenType.RPAREN,
                TokenType.RBRACKET,
                TokenType.RBRACE,
                TokenType.COMMA,
                TokenType.SEMICOLON,
            ):
                break

            precedence = self.get_precedence(token.type)
            if precedence < min_precedence:
                break

            # Handle assignment (right-associative)
            if token.type == TokenType.ASSIGN:
                if not isinstance(left, (IdentifierNode, ArrayIndexNode)):
                    raise SyntaxError(
                        "Can only assign to identifiers or array elements"
                    )

                self.advance()
                right = self.parse_binary_expression(
                    self.parse_primary(), precedence - 1
                )
                left = AssignmentNode(left=left, right=right)
                continue

            # Handle other binary operators
            match token.type:
                case (
                    TokenType.PLUS
                    | TokenType.MINUS
                    | TokenType.STAR
                    | TokenType.SLASH
                    | TokenType.MOD
                    | TokenType.EQ
                    | TokenType.NEQ
                    | TokenType.LT
                    | TokenType.GT
                    | TokenType.LTE
                    | TokenType.GTE
                    | TokenType.AND
                    | TokenType.OR
                ):
                    operator = token.value if token.value else token.type.name
                    self.advance()

                    # Parse right operand with higher precedence
                    right = self.parse_binary_expression(
                        self.parse_primary(), precedence + 1
                    )
                    left = BinaryOpNode(left=left, operator=operator, right=right)

                case _:
                    break

        return left

    def parse_expression(self) -> ASTNode:
        """Parse an expression."""
        left = self.parse_primary()
        left = self.parse_postfix(left)
        return self.parse_binary_expression(left)

    def parse_block(self) -> BlockNode:
        """Parse a block of statements: { statement* }"""
        self.expect(TokenType.LBRACE)
        statements: List[ASTNode] = []

        while (
            self.current.type != TokenType.RBRACE and self.current.type != TokenType.EOF
        ):
            statements.append(self.parse_statement())

        self.expect(TokenType.RBRACE)
        return BlockNode(statements=statements)

    def parse_variable_declaration(self) -> ASTNode:
        """Parse variable declaration: type identifier (= expression)? ;"""
        var_type = self.parse_type()
        is_array = var_type == SymbolType.INT_ARRAY

        # Parse identifier
        var_name_token = self.expect(TokenType.IDENTIFIER, "Expected variable name")
        var_name = var_name_token.value

        # Check for array dimensions
        dimensions = 0
        while self.current.type == TokenType.LBRACKET:
            self.advance()
            # Optional array size: allow an integer literal inside brackets
            if self.current.type == TokenType.INTEGER:
                # consume the size token (we don't currently store it)
                self.advance()
            self.expect(TokenType.RBRACKET)
            dimensions += 1

        # Check for initialization
        init_value = None
        if self.match(TokenType.ASSIGN):
            init_value = self.parse_expression()

        self.expect(TokenType.SEMICOLON)

        # Declare in symbol table
        self.symbol_table.declare(
            name=var_name,
            type_=var_type,
            is_array=is_array or dimensions > 0,
            value=None,
        )

        return VariableDeclarationNode(
            var_name=var_name, var_type=var_type, init_value=init_value
        )

    def parse_function_declaration(self) -> FunctionDeclarationNode:
        """Parse function declaration/definition: type ident '(' params ')' '{' body '}'"""
        # Parse return type
        return_type = self.parse_type()

        # Function name
        name_token = self.expect(TokenType.IDENTIFIER, "Expected function name")
        func_name = name_token.value

        # Parameters
        self.expect(TokenType.LPAREN)
        arg_types: List[SymbolType] = []
        arg_names: List[str] = []

        if self.current.type != TokenType.RPAREN:
            # At least one parameter
            while True:
                param_type = self.parse_type()
                param_name_token = self.expect(
                    TokenType.IDENTIFIER, "Expected parameter name"
                )
                arg_types.append(param_type)
                arg_names.append(param_name_token.value)

                if self.current.type == TokenType.COMMA:
                    self.advance()
                    continue
                break

        self.expect(TokenType.RPAREN)

        # Register function in symbol table (store parameter types)
        sym = self.symbol_table.declare(
            name=func_name, type_=return_type, is_array=False, is_function=True
        )
        sym.parameters = arg_types

        # If next token is semicolon, this is a prototype/declaration without body
        if self.current.type == TokenType.SEMICOLON:
            self.advance()
            return FunctionDeclarationNode(
                func_name=func_name,
                return_type=return_type,
                arg_types=arg_types,
                arg_names=arg_names,
                body=None,
            )

        # Otherwise expect a body block. Create a new scope for parameters and body
        outer_table = self.symbol_table
        self.symbol_table = SymbolTable(parent=outer_table)

        # Declare parameters in function-local scope
        for pname, ptype in zip(arg_names, arg_types):
            self.symbol_table.declare(
                name=pname, type_=ptype, is_array=(ptype == SymbolType.INT_ARRAY)
            )

        # Parse function body within the new scope
        body = self.parse_block()

        # Restore outer symbol table
        self.symbol_table = outer_table

        return FunctionDeclarationNode(
            func_name=func_name,
            return_type=return_type,
            arg_types=arg_types,
            arg_names=arg_names,
            body=body,
        )

    def parse_return_statement(self) -> ReturnStatementNode:
        """Parse return statement: return expr? ;"""
        self.expect(TokenType.RETURN)
        expr = None
        if self.current.type != TokenType.SEMICOLON:
            expr = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return ReturnStatementNode(expression=expr)

    def parse_if_statement(self) -> IfStatementNode:
        """Parse if statement: if (expr) { ... } else { ... }"""
        self.expect(TokenType.IF)
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # Parse then block
        then_block = self.parse_statement()

        # Parse else block if present
        else_block = None
        if self.match(TokenType.ELSE):
            else_block = self.parse_statement()

        return IfStatementNode(
            condition=condition, then_block=then_block, else_block=else_block
        )

    def parse_while_statement(self) -> WhileStatementNode:
        """Parse while statement: while (expr) { ... }"""
        self.expect(TokenType.WHILE)
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # Parse loop body
        body = self.parse_statement()

        return WhileStatementNode(condition=condition, body=body)

    def parse_statement(self) -> ASTNode:
        """Parse a statement."""
        match self.current.type:
            case TokenType.INT_TYPE | TokenType.BOOL_TYPE | TokenType.VOID_TYPE:
                # Lookahead: if pattern is 'type ident (' it's a function declaration
                next_type = (
                    self.tokens[self.pos + 1].type
                    if self.pos + 1 < len(self.tokens)
                    else TokenType.EOF
                )
                next2_type = (
                    self.tokens[self.pos + 2].type
                    if self.pos + 2 < len(self.tokens)
                    else TokenType.EOF
                )
                if next_type == TokenType.IDENTIFIER and next2_type == TokenType.LPAREN:
                    return self.parse_function_declaration()
                return self.parse_variable_declaration()

            case TokenType.IF:
                return self.parse_if_statement()

            case TokenType.WHILE:
                return self.parse_while_statement()

            case TokenType.LBRACE:
                return self.parse_block()
            case TokenType.RETURN:
                return self.parse_return_statement()

            case _:
                # Expression statement
                expr = self.parse_expression()
                self.expect(TokenType.SEMICOLON, "Expected semicolon after expression")
                return ExpressionStatementNode(expression=expr)

    def parse_program(self) -> ProgramNode:
        """Parse a complete program (sequence of statements)."""
        statements: List[ASTNode] = []

        while self.current.type != TokenType.EOF:
            statements.append(self.parse_statement())

        return ProgramNode(statements=statements)

    def parse(self) -> ASTNode:
        """Parse complete expression or statement."""
        # Check if we have a full program (multiple statements)
        program_tokens = {
            TokenType.IF,
            TokenType.WHILE,
            TokenType.LBRACE,
            TokenType.INT_TYPE,
            TokenType.BOOL_TYPE,
        }

        if any(token.type in program_tokens for token in self.tokens):
            return self.parse_program()
        else:
            # Try to parse as expression or single statement
            result = self.parse_statement()
            if self.current.type != TokenType.EOF:
                raise SyntaxError(f"Unexpected tokens at end: {self.current}")
            return result
