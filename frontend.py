import re

# Token types
INTEGER = 'INTEGER'
IDENTIFIER = 'IDENTIFIER'
PLUS = 'PLUS'
MINUS = 'MINUS'
STAR = 'STAR'
MOD = 'MOD'
SLASH = 'SLASH'
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'
LBRACKET = 'LBRACKET'
RBRACKET = 'RBRACKET'
LBRACE = 'LBRACE'
RBRACE = 'RBRACE'
COMMA = 'COMMA'
ASSIGN = 'ASSIGN'
SEMICOLON = 'SEMICOLON'
EQ = 'EQ'
NEQ = 'NEQ'
LT = 'LT'
GT = 'GT'
LTE = 'LTE'
GTE = 'GTE'
AND = 'AND'
OR = 'OR'
NOT = 'NOT'
WHILE = 'WHILE'
IF = 'IF'
ELSE = 'ELSE'
INT_TYPE = 'INT_TYPE'
BOOL_TYPE = 'BOOL_TYPE'
TRUE = 'TRUE'
FALSE = 'FALSE'
EOF = 'EOF'

# AST node types
class NodeTypes:
    INT_LITERAL = 'int_literal'
    BOOL_LITERAL = 'bool_literal'
    IDENTIFIER = 'identifier'
    BINARY_OP = 'binary_op'
    UNARY_OP = 'unary_op'
    ARRAY_INDEX = 'array_index'
    FUNC_CALL = 'func_call'
    ASSIGNMENT = 'assignment'
    EXPR_STMT = 'expr_stmt'
    WHILE_STMT = 'while_stmt'
    IF_STMT = 'if_stmt'
    BLOCK = 'block'
    VAR_DECL = 'var_decl'
    FUNC_DECL = 'func_decl'
    PROGRAM = 'program'

# Token class
class Token:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value
    
    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)})"

# Lexer implementation
class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None
        self.keywords = {
            'while': WHILE,
            'if': IF,
            'else': ELSE,
            'int': INT_TYPE,
            'bool': BOOL_TYPE,
            'true': TRUE,
            'false': FALSE
        }
    
    def error(self):
        raise SyntaxError(f'Invalid character at position {self.pos}: {self.current_char}')
    
    def advance(self):
        """Advance to next character."""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None
    
    def skip_whitespace(self):
        """Skip whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self):
        """Skip single-line comments (// ...)."""
        if self.current_char == '/' and self.peek_char() == '/':
            while self.current_char is not None and self.current_char != '\n':
                self.advance()
            if self.current_char == '\n':
                self.advance()
    
    def peek_char(self):
        """Look at next character without consuming it."""
        next_pos = self.pos + 1
        if next_pos < len(self.text):
            return self.text[next_pos]
        return None
    
    def integer(self):
        """Parse a multi-digit integer."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)
    
    def identifier(self):
        """Parse an identifier or keyword."""
        result = ''
        # First character must be letter or underscore
        if self.current_char is not None and (self.current_char.isalpha() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        # Following characters can be letters, digits, or underscores
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        return result
    
    def get_next_token(self):
        """Lexical analyzer that returns tokens one at a time."""
        while self.current_char is not None:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Skip comments
            if self.current_char == '/' and self.peek_char() == '/':
                self.skip_comment()
                continue
            
            # Integers
            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())
            
            # Identifiers and keywords
            if self.current_char.isalpha() or self.current_char == '_':
                ident = self.identifier()
                token_type = self.keywords.get(ident, IDENTIFIER)
                return Token(token_type, ident)
            
            # Operators and punctuation
            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')
            
            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')
            
            if self.current_char == '*':
                self.advance()
                return Token(STAR, '*')
            
            if self.current_char == '%':
                self.advance()
                return Token(MOD, '%')
            
            if self.current_char == '/':
                self.advance()
                return Token(SLASH, '/')
            
            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')
            
            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')
            
            if self.current_char == '[':
                self.advance()
                return Token(LBRACKET, '[')
            
            if self.current_char == ']':
                self.advance()
                return Token(RBRACKET, ']')
            
            if self.current_char == '{':
                self.advance()
                return Token(LBRACE, '{')
            
            if self.current_char == '}':
                self.advance()
                return Token(RBRACE, '}')
            
            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')
            
            if self.current_char == ';':
                self.advance()
                return Token(SEMICOLON, ';')
            
            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(EQ, '==')
                return Token(ASSIGN, '=')
            
            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(LTE, '<=')
                return Token(LT, '<')
            
            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(GTE, '>=')
                return Token(GT, '>')
            
            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(NEQ, '!=')
                return Token(NOT, '!')
            
            if self.current_char == '&':
                self.advance()
                if self.current_char == '&':
                    self.advance()
                    return Token(AND, '&&')
                self.error()
            
            if self.current_char == '|':
                self.advance()
                if self.current_char == '|':
                    self.advance()
                    return Token(OR, '||')
                self.error()
            
            self.error()
        
        return Token(EOF, None)
    
    def tokenize(self):
        """Return all tokens from the input string."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == EOF:
                break
        return tokens
    
# Symbol table entry
class Symbol:
    def __init__(self, name, type_, is_array=False, is_function=False, value=None):
        self.name = name
        self.type = type_  # 'int', 'bool', 'int[]', or function type
        self.is_array = is_array
        self.is_function = is_function
        self.value = value
    
    def __repr__(self):
        return f"Symbol({self.name}, {self.type}, is_array={self.is_array}, is_function={self.is_function})"
    
# Symbol Table
class SymbolTable:
    def __init__(self):
        self.symbols = {}  # name -> Symbol
    
    def declare(self, name, type_, is_array=False, is_function=False, value=None):
        """Declare a new variable."""
        if name in self.symbols:
            raise SyntaxError(f"Variable '{name}' already declared")
        self.symbols[name] = Symbol(name, type_, is_array, is_function, value)
    
    def lookup(self, name):
        """Look up a variable."""
        if name not in self.symbols:
            raise SyntaxError(f"Undeclared variable '{name}'")
        return self.symbols[name]
    
    def exists(self, name):
        """Check if variable is declared."""
        return name in self.symbols
    
class SymbolType:
    INT = 'int'
    BOOL = 'bool'
    INT_ARRAY = 'int[]'

# Parser implementation (Pratt parsing)
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else Token(EOF, None)
        self.symbol_table = SymbolTable()
        
        # Operator precedence table (higher = tighter binding)
        self.precedence = {
            ASSIGN: 0,
            OR: 1,
            AND: 2,
            EQ: 3, NEQ: 3,
            LT: 4, GT: 4, LTE: 4, GTE: 4,
            PLUS: 5, MINUS: 5,
            STAR: 6, SLASH: 6, MOD: 6,
            LBRACKET: 7, LPAREN: 7,  # For indexing and function calls
            NOT: 8  # Unary operators have highest precedence
        }
    
    def peek(self):
        """Return next token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(EOF, None)
    
    def advance(self):
        """Move to next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = Token(EOF, None)
        return self.current
    
    def expect(self, type_):
        """Expect and consume token of given type."""
        if self.current.type == type_:
            token = self.current
            self.advance()
            return token
        raise SyntaxError(f"Expected {type_}, got {self.current.type}")
    
    def match(self, type_):
        """Check if current token matches type, consume if true."""
        if self.current.type == type_:
            self.advance()
            return True
        return False
    
    def get_precedence(self, token_type):
        """Get precedence for operator token type."""
        return self.precedence.get(token_type, 0)
    
    def parse_type(self):
        """Parse a type: int, bool, int[]."""
        if self.current.type == INT_TYPE:
            self.advance()
            # Check for array type: int[]
            if self.current.type == LBRACKET:
                self.advance()
                self.expect(RBRACKET)
                return SymbolType.INT_ARRAY
            return SymbolType.INT
        elif self.current.type == BOOL_TYPE:
            self.advance()
            return SymbolType.BOOL
        else:
            raise SyntaxError(f"Expected type, got {self.current.type}")
    
    def parse_prefix(self):
        """Parse prefix expressions."""
        token = self.current
        
        if token.type == INTEGER:
            self.advance()
            return {'type': NodeTypes.INT_LITERAL, 'value': token.value}
        
        elif token.type in (TRUE, FALSE):
            self.advance()
            return {'type': NodeTypes.BOOL_LITERAL, 'value': token.type == TRUE}
        
        elif token.type == IDENTIFIER:
            # Check if variable is declared
            var_name = token.value
            if not self.symbol_table.exists(var_name):
                raise SyntaxError(f"Undeclared variable '{var_name}'")
            symbol = self.symbol_table.lookup(var_name)
            self.advance()
            return {
                'type': NodeTypes.IDENTIFIER, 
                'name': var_name,
                'symbol_type': symbol.type,
                'is_array': symbol.is_array
            }
        
        elif token.type == LPAREN:
            self.advance()  # Consume '('
            expr = self.parse_expression()
            self.expect(RPAREN)
            return expr
        
        elif token.type == MINUS:  # Unary minus
            self.advance()
            right = self.parse_prefix()
            return {'type': NodeTypes.UNARY_OP, 'operator': '-', 'right': right}
        
        elif token.type == NOT:  # Unary not
            self.advance()
            right = self.parse_prefix()
            return {'type': NodeTypes.UNARY_OP, 'operator': '!', 'right': right}
        
        elif token.type == LBRACE:  # Block as expression (for completeness)
            return self.parse_block()
        
        raise SyntaxError(f"Unexpected token in prefix: {token}")
    
    def parse_infix(self, left, min_precedence):
        """Parse infix/postfix expressions."""
        token = self.current
        
        # Binary operators
        if token.type in (PLUS, MINUS, STAR, MOD, SLASH, EQ, NEQ, LT, GT, LTE, GTE, AND, OR):
            op_token = token
            self.advance()
            right = self.parse_expression(min_precedence)
            return {
                'type': NodeTypes.BINARY_OP,
                'left': left,
                'operator': op_token.value if op_token.value else op_token.type,
                'right': right
            }
        
        # Assignment (right-associative)
        elif token.type == ASSIGN:
            # Check if left side is assignable (identifier or array index)
            if left['type'] not in [NodeTypes.IDENTIFIER, NodeTypes.ARRAY_INDEX]:
                raise SyntaxError(f"Cannot assign to {left['type']}")
            
            self.advance()
            right = self.parse_expression(min_precedence - 1)  # Right associative
            return {
                'type': NodeTypes.ASSIGNMENT,
                'left': left,
                'right': right
            }
        
        # Array indexing
        elif token.type == LBRACKET:
            # Check if left is an array
            if left['type'] != NodeTypes.IDENTIFIER or not left.get('is_array', False):
                raise SyntaxError(f"Cannot index non-array type")
            
            self.advance()
            index = self.parse_expression()
            self.expect(RBRACKET)
            return {
                'type': NodeTypes.ARRAY_INDEX,
                'array': left,
                'index': index
            }
        
        # Function call
        elif token.type == LPAREN:
            self.advance()
            args = []
            
            # Parse arguments if present
            if self.current.type != RPAREN:
                args.append(self.parse_expression())
                while self.current.type == COMMA:
                    self.advance()
                    args.append(self.parse_expression())
            
            self.expect(RPAREN)
            return {
                'type': NodeTypes.FUNC_CALL,
                'function': left,
                'arguments': args
            }
        
        raise SyntaxError(f"Unexpected token in infix: {token}")
    
    def parse_expression(self, min_precedence=0):
        """Main Pratt parser entry point."""
        # Parse left operand
        left = self.parse_prefix()
        
        # Parse following operators with higher precedence
        while (self.current.type not in (EOF, RPAREN, RBRACKET, RBRACE, COMMA, SEMICOLON) and 
               self.get_precedence(self.current.type) >= min_precedence):
            
            token_type = self.current.type
            
            # For right-associative operators (assignment)
            if token_type == ASSIGN:
                next_prec = self.get_precedence(token_type)
                left = self.parse_infix(left, next_prec)
            else:
                next_prec = self.get_precedence(token_type)
                left = self.parse_infix(left, next_prec + 1)
        
        return left
    
    def parse_block(self):
        """Parse a block of statements: { statement* }"""
        self.expect(LBRACE)
        statements = []
        
        while self.current.type != RBRACE and self.current.type != EOF:
            statements.append(self.parse_statement())
        
        self.expect(RBRACE)
        return {'type': NodeTypes.BLOCK, 'statements': statements}
    
    def parse_var_declaration(self):
        """Parse variable declaration: type identifier (= expression)? ;"""
        start_pos = self.pos
        
        # Parse type
        var_type = self.parse_type()
        is_array = var_type == SymbolType.INT_ARRAY 
        
        # Could be function signature
        is_function = False
        args_types = []
        
        # Parse identifier
        var_name = self.expect(IDENTIFIER).value
        
        # Check for optional initialization
        init_value = None
        if self.match(ASSIGN):
            init_value = self.parse_expression()
        elif self.current.type == LPAREN:
            self.advance()
            while self.current.type != RPAREN:
                args_types.append(self.parse_type())
                if self.current.type == COMMA:
                    self.advance()
            self.expect(RPAREN)
            is_function = True
        
        self.symbol_table.declare(var_name, var_type, is_array=is_array, is_function=is_function)
        self.expect(SEMICOLON)
        
        # Create AST node
        if is_function:
            node = {
                'type': NodeTypes.FUNC_DECL,
                'func_name': var_name,
                'return_type': var_type,
                'arg_types': args_types
            }
        else:
          node = {
                'type': NodeTypes.VAR_DECL,
                'var_name': var_name,
                'var_type': var_type,
                'init_value': init_value
            }
        
        return node
    
    def parse_statement(self):
        """Parse a statement."""
        token = self.current
        
        # Variable declaration (int, bool, or int[])
        if token.type in (INT_TYPE, BOOL_TYPE):
            return self.parse_var_declaration()
        
        # If statement
        if token.type == IF:
            return self.parse_if_statement()
        
        # While statement
        elif token.type == WHILE:
            return self.parse_while_statement()
        
        # Block
        elif token.type == LBRACE:
            return self.parse_block()
        
        # Expression statement (including assignment)
        else:
            expr = self.parse_expression()
            
            # Check if this is an assignment statement
            if self.current.type == SEMICOLON:
                self.advance()
                # If it's an assignment expression, keep it as assignment
                if expr['type'] == NodeTypes.ASSIGNMENT:
                    # Verify left side is declared
                    if expr['left']['type'] == NodeTypes.IDENTIFIER:
                        var_name = expr['left']['name']
                        if not self.symbol_table.exists(var_name):
                            raise SyntaxError(f"Assignment to undeclared variable '{var_name}'")
                    return expr
                # Otherwise, wrap in expression statement
                else:
                    return {'type': NodeTypes.EXPR_STMT, 'expression': expr}
            
            # If no semicolon, might be inside expression context
            return expr
    
    def parse_if_statement(self):
        """Parse if statement: if (expr) { ... } else { ... }"""
        self.expect(IF)
        self.expect(LPAREN)
        condition = self.parse_expression()
        self.expect(RPAREN)
        
        # Parse then block
        then_block = self.parse_statement()
        
        # Parse else block if present
        else_block = None
        if self.match(ELSE):
            else_block = self.parse_statement()
        
        return {
            'type': NodeTypes.IF_STMT,
            'condition': condition,
            'then_block': then_block,
            'else_block': else_block
        }
    
    def parse_while_statement(self):
        """Parse while statement: while (expr) { ... }"""
        self.expect(WHILE)
        self.expect(LPAREN)
        condition = self.parse_expression()
        self.expect(RPAREN)
        
        # Parse loop body
        body = self.parse_statement()
        
        return {
            'type': NodeTypes.WHILE_STMT,
            'condition': condition,
            'body': body
        }
    
    def parse_program(self):
        """Parse a complete program (sequence of statements)."""
        statements = []
        
        while self.current.type != EOF:
            statements.append(self.parse_statement())
        
        return {
            'type': NodeTypes.PROGRAM,
            'statements': statements,
            'symbol_table': self.symbol_table
        }
    
    def parse(self):
        """Parse complete expression or statement."""
        # Check if we have a full program (multiple statements)
        if any(token.type in (IF, WHILE, LBRACE, INT_TYPE, BOOL_TYPE) for token in self.tokens):
            return self.parse_program()
        else:
            # Try to parse as expression or single statement
            result = self.parse_statement()
            if self.current.type != EOF:
                raise SyntaxError(f"Unexpected tokens at end: {self.current}")
            return result

# Convenience functions
def lex(text):
    """Tokenize input string."""
    lexer = Lexer(text)
    return lexer.tokenize()

def parse_tokens(tokens):
    """Parse tokens into AST."""
    parser = Parser(tokens)
    return parser.parse()

def parse_string(text):
    """Parse string directly into AST."""
    tokens = lex(text)
    return parse_tokens(tokens)

# Pretty printer for AST
def print_ast(node, indent=0):
    """Pretty print AST."""
    if not isinstance(node, dict):
        print(" " * indent + str(node))
        return
    
    node_type = node.get('type', '')
    indent_str = " " * indent
    
    if node_type == NodeTypes.INT_LITERAL:
        print(f"{indent_str}IntLiteral({node['value']})")
    
    elif node_type == NodeTypes.BOOL_LITERAL:
        print(f"{indent_str}BoolLiteral({node['value']})")
    
    elif node_type == NodeTypes.IDENTIFIER:
        type_info = f", type={node.get('symbol_type', 'unknown')}"
        if node.get('is_array'):
            type_info += " (array)"
        print(f"{indent_str}Identifier({node['name']}{type_info})")
    
    elif node_type == NodeTypes.BINARY_OP:
        print(f"{indent_str}BinaryOp({node['operator']})")
        print_ast(node['left'], indent + 2)
        print_ast(node['right'], indent + 2)
    
    elif node_type == NodeTypes.UNARY_OP:
        print(f"{indent_str}UnaryOp({node['operator']})")
        print_ast(node['right'], indent + 2)
    
    elif node_type == NodeTypes.ARRAY_INDEX:
        print(f"{indent_str}ArrayIndex")
        print_ast(node['array'], indent + 2)
        print_ast(node['index'], indent + 2)
    
    elif node_type == NodeTypes.FUNC_CALL:
        print(f"{indent_str}FunctionCall({node['function'].get('name', 'anonymous')})")
        for i, arg in enumerate(node['arguments']):
            print(f"{indent_str + (' ' * 2)}Arg[{i}]:")
            print_ast(arg, indent + 4)
    
    elif node_type == NodeTypes.ASSIGNMENT:
        print(f"{indent_str}Assignment")
        print_ast(node['left'], indent + 2)
        print_ast(node['right'], indent + 2)
    
    elif node_type == NodeTypes.EXPR_STMT:
        print(f"{indent_str}ExpressionStatement")
        print_ast(node['expression'], indent + 2)
    
    elif node_type == NodeTypes.WHILE_STMT:
        print(f"{indent_str}WhileStatement")
        print(f"{indent_str + ' ' * 2}Condition:")
        print_ast(node['condition'], indent + 4)
        print(f"{indent_str + ' ' * 2}Body:")
        print_ast(node['body'], indent + 4)
    
    elif node_type == NodeTypes.IF_STMT:
        print(f"{indent_str}IfStatement")
        print(f"{indent_str + ' ' * 2}Condition:")
        print_ast(node['condition'], indent + 4)
        print(f"{indent_str + ' ' * 2}Then:")
        print_ast(node['then_block'], indent + 4)
        if node.get('else_block'):
            print(f"{indent_str + ' ' * 2}Else:")
            print_ast(node['else_block'], indent + 4)
    
    elif node_type == NodeTypes.BLOCK:
        print(f"{indent_str}Block")
        for i, stmt in enumerate(node['statements']):
            print(f"{indent_str + ' ' * 2}Statement[{i}]:")
            print_ast(stmt, indent + 4)
    
    elif node_type == NodeTypes.VAR_DECL:
        init_str = f" = ..." if node['init_value'] else ""
        print(f"{indent_str}VarDecl({node['var_name']}: {node['var_type']}{init_str})")
        if node['init_value']:
            print_ast(node['init_value'], indent + 2)
            
    elif node_type == NodeTypes.FUNC_DECL:
        print(f"{indent_str}FuncDecl({node['func_name']}: {node['return_type']}, args={node['arg_types']})")
    
    elif node_type == NodeTypes.PROGRAM:
        print(f"{indent_str}Program")
        print(f"{' ' * (indent + 2)}Symbol Table:")
        for name, symbol in node.get('symbol_table', SymbolTable()).symbols.items():
            print(f"{' ' * (indent + 4)}{symbol}")
        print(f"{' ' * (indent + 2)}Statements:")
        for i, stmt in enumerate(node['statements']):
            print(f"{' ' * (indent + 4)}Statement[{i}]:")
            print_ast(stmt, indent + 6)
    
    else:
        print(f"{indent_str}Unknown node type: {node_type}")

# Type checker helper
def check_type_compatibility(node, expected_type=None):
    """Simple type checking for expressions."""
    if not isinstance(node, dict):
        return expected_type if expected_type else 'unknown'
    
    node_type = node.get('type', '')
    
    if node_type == NodeTypes.INT_LITERAL:
        return 'int'
    
    elif node_type == NodeTypes.BOOL_LITERAL:
        return 'bool'
    
    elif node_type == NodeTypes.IDENTIFIER:
        return node.get('symbol_type', 'unknown')
    
    elif node_type == NodeTypes.BINARY_OP:
        op = node['operator']
        left_type = check_type_compatibility(node['left'])
        right_type = check_type_compatibility(node['right'])
        
        # Arithmetic operators
        if op in ('+', '-', '*', '/'):
            if left_type == 'int' and right_type == 'int':
                return 'int'
            raise SyntaxError(f"Cannot apply '{op}' to types '{left_type}' and '{right_type}'")
        
        # Comparison operators
        elif op in ('==', '!=', '<', '>', '<=', '>='):
            if left_type == right_type:
                return 'bool'
            raise SyntaxError(f"Cannot compare types '{left_type}' and '{right_type}'")
        
        # Logical operators
        elif op in ('&&', '||'):
            if left_type == 'bool' and right_type == 'bool':
                return 'bool'
            raise SyntaxError(f"Cannot apply logical '{op}' to non-boolean types")
    
    elif node_type == NodeTypes.UNARY_OP:
        op = node['operator']
        expr_type = check_type_compatibility(node['right'])
        
        if op == '-' and expr_type == 'int':
            return 'int'
        elif op == '!' and expr_type == 'bool':
            return 'bool'
        raise SyntaxError(f"Cannot apply unary '{op}' to type '{expr_type}'")
    
    elif node_type == NodeTypes.ARRAY_INDEX:
        array_type = node['array'].get('symbol_type', 'unknown')
        if array_type != 'int[]':
            raise SyntaxError(f"Cannot index non-array type '{array_type}'")
        index_type = check_type_compatibility(node['index'])
        if index_type != 'int':
            raise SyntaxError(f"Array index must be integer, got '{index_type}'")
        return 'int'
    
    return 'unknown'

# Test the complete pipeline
if __name__ == "__main__":
    test_cases = [
        # Simple assignment statement
        "int x = 5;",
        # While loop
        "int i; while (i < 10) { i = i + 1; }",
        # If statement
        "int x; int y; if (x > 0) { y = 1; } else { y = -1; }",
        # Complex program
        """
        int i = 0;
        int sum = 0;
        while (i < 10) {
            if (i % 2 == 0) {
                sum = sum + i;
            }
            i = i + 1;
        }
        """,
        # Array operations
        "int[] arr; arr[0] = 5;",
        # Function call in loop
        """
        int value;
        int hasNext();
        int getNext();
        int process(int);
        while (hasNext()) {
            value = getNext();
            process(value);
        }
        """,
        # Nested if-else
        """
        int a; int b; int max = 0;
        if (a > b) {
            max = a;
        } else if (a < b) {
            max = b;
        } else {
            max = 0;
        }
        """
    ]
    
    print("Testing lexer and parser with statements:")
    print("=" * 80)
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print("-" * 40)
        print(f"Input:\n{test.strip()}")
        print("-" * 40)
        
        try:
            # Tokenize
            tokens = lex(test)
            print(f"Tokens: {tokens}")
            
            # Parse
            ast = parse_tokens(tokens)
            print("\nAST:")
            print_ast(ast)
            
            # Type check declarations
            if ast['type'] == NodeTypes.PROGRAM:
                for stmt in ast['statements']:
                    if stmt['type'] == NodeTypes.VAR_DECL and stmt['init_value']:
                        try:
                            init_type = check_type_compatibility(stmt['init_value'])
                            var_type = stmt['var_type']
                            if init_type != var_type and not (var_type == 'int[]' and init_type == 'int[]'):
                                print(f"Warning: Type mismatch in initialization of {stmt['var_name']}: "
                                      f"expected {var_type}, got {init_type}")
                        except SyntaxError as e:
                            print(f"Type error: {e}")
            
        except SyntaxError as e:
            print(f"Error: {e}")
        
        print()

    
    
    while True:
        # Interactive mode
        print("\nInteractive mode (type 'quit' to exit):")
        print("=" * 80)
        try:
            text = input("\nEnter program or expression: ").strip()
            if text.lower() in ('quit', 'exit', 'q'):
                break
            
            if not text:
                continue
            
            # Tokenize
            tokens = lex(text)
            print(f"\nTokens: {tokens}")
            
            # Parse
            ast = parse_tokens(tokens)
            print("\nAST structure:")
            print_ast(ast)
            
        except SyntaxError as e:
            print(f"Syntax error: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")