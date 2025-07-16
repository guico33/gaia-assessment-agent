"""
Comprehensive tests for SafeCodeExecutor class.
"""

import pytest

from src.code_executor import SafeCodeExecutor


class TestSafeCodeExecutor:
    """Test suite for SafeCodeExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = SafeCodeExecutor()

    # Basic functionality tests
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        result = self.executor.execute("print(2 + 2)")
        assert "Code executed successfully:" in result
        assert "4" in result

    def test_basic_print(self):
        """Test basic print functionality."""
        result = self.executor.execute("print('Hello, World!')")
        assert "Code executed successfully:" in result
        assert "Hello, World!" in result

    def test_no_output_code(self):
        """Test code that produces no output."""
        result = self.executor.execute("x = 5")
        assert "Code executed successfully (no output)" in result

    # Built-in functions tests
    def test_safe_builtins(self):
        """Test that safe built-in functions work."""
        test_cases = [
            ("print(abs(-5))", "5"),
            ("print(max([1, 2, 3]))", "3"),
            ("print(min([1, 2, 3]))", "1"),
            ("print(sum([1, 2, 3]))", "6"),
            ("print(len([1, 2, 3]))", "3"),
            ("print(round(3.14159, 2))", "3.14"),
            ("print(str(42))", "42"),
            ("print(int('42'))", "42"),
            ("print(float('3.14'))", "3.14"),
        ]

        for code, expected in test_cases:
            result = self.executor.execute(code)
            assert "Code executed successfully:" in result
            assert expected in result

    # Module import tests
    def test_allowed_module_imports(self):
        """Test that allowed modules can be imported."""
        allowed_modules = [
            "math",
            "statistics",
            "json",
            "datetime",
            "re",
            "random",
            "itertools",
            "functools",
            "collections",
            "operator",
        ]

        for module in allowed_modules:
            result = self.executor.execute(
                f"import {module}\nprint('imported {module}')"
            )
            assert "Code executed successfully:" in result
            assert f"imported {module}" in result

    def test_direct_module_access(self):
        """Test direct access to pre-imported modules."""
        test_cases = [
            ("print(math.pi)", "3.14159"),
            ("print(math.sqrt(16))", "4.0"),
            ("print(statistics.mean([1, 2, 3]))", "2"),
            ("print(json.dumps({'key': 'value'}))", '{"key": "value"}'),
        ]

        for code, expected in test_cases:
            result = self.executor.execute(code)
            assert "Code executed successfully:" in result
            assert expected in result

    # Security tests
    def test_dangerous_imports_blocked(self):
        """Test that dangerous imports are blocked."""
        dangerous_imports = [
            "import os",
            "import sys",
            "from os import getcwd",
            "from sys import exit",
            "import subprocess",
            "import importlib",
        ]

        for dangerous_code in dangerous_imports:
            result = self.executor.execute(dangerous_code)
            assert "Error: Code contains potentially dangerous operations" in result

    def test_dangerous_functions_blocked(self):
        """Test that dangerous functions are blocked."""
        dangerous_functions = [
            "exec('print(1)')",
            "eval('2 + 2')",
            "open('test.txt')",
            "compile('print(1)', 'test', 'exec')",
            "globals()",
            "locals()",
            "getattr(object, 'method')",
            "setattr(object, 'attr', 'value')",
            "delattr(object, 'attr')",
            "vars()",
            "__import__('os')",
        ]

        for dangerous_code in dangerous_functions:
            result = self.executor.execute(dangerous_code)
            assert "Error: Code contains potentially dangerous operations" in result

    def test_system_access_blocked(self):
        """Test that system access patterns are blocked."""
        system_access_patterns = [
            "os.getcwd()",
            "sys.exit()",
            "__file__",
            "__name__",
            "__builtins__",
        ]

        for pattern in system_access_patterns:
            result = self.executor.execute(pattern)
            assert "Error: Code contains potentially dangerous operations" in result

    # Error handling tests
    def test_syntax_error_handling(self):
        """Test syntax error handling."""
        result = self.executor.execute("print(")
        assert "Syntax Error:" in result

    def test_name_error_handling(self):
        """Test name error handling."""
        result = self.executor.execute("print(undefined_variable)")
        assert "Name Error:" in result

    def test_type_error_handling(self):
        """Test type error handling."""
        result = self.executor.execute("print(1 + '2')")
        assert "Type Error:" in result

    def test_value_error_handling(self):
        """Test value error handling."""
        result = self.executor.execute("print(int('not_a_number'))")
        assert "Value Error:" in result

    def test_zero_division_error_handling(self):
        """Test zero division error handling."""
        result = self.executor.execute("print(1 / 0)")
        assert "Division by Zero Error:" in result

    # Complex computation tests
    def test_mathematical_computation(self):
        """Test complex mathematical computations."""
        code = """
import math
result = math.sqrt(16) + math.sin(math.pi/2) + math.factorial(5)
print(f"Result: {result}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "Result: 125.0" in result

    def test_data_processing(self):
        """Test data processing capabilities."""
        code = """
import statistics
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean = statistics.mean(data)
median = statistics.median(data)
stdev = round(statistics.stdev(data), 2)
print(f"Mean: {mean}, Median: {median}, StdDev: {stdev}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "Mean: 5.5" in result
        assert "Median: 5.5" in result

    def test_string_processing(self):
        """Test string processing with regex."""
        code = """
import re
text = "The year 2024 was great!"
years = re.findall(r'\\d{4}', text)
print(f"Found years: {years}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "Found years: ['2024']" in result

    def test_collections_usage(self):
        """Test collections module usage."""
        code = """
from collections import Counter
items = ['apple', 'banana', 'apple', 'cherry', 'banana', 'banana']
counter = Counter(items)
print(f"Most common: {counter.most_common(1)}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "Most common: [('banana', 3)]" in result

    # Edge cases
    def test_empty_code(self):
        """Test execution of empty code."""
        result = self.executor.execute("")
        assert "Code executed successfully (no output)" in result

    def test_whitespace_only_code(self):
        """Test execution of whitespace-only code."""
        result = self.executor.execute("   \n  \t  \n  ")
        assert "Code executed successfully (no output)" in result

    def test_comment_only_code(self):
        """Test execution of comment-only code."""
        result = self.executor.execute("# This is just a comment")
        assert "Code executed successfully (no output)" in result

    def test_multiline_code(self):
        """Test execution of multiline code."""
        code = """
x = 10
y = 20
z = x + y
print(f"Sum: {z}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "Sum: 30" in result

    def test_case_insensitive_security_check(self):
        """Test that security checks are case-insensitive."""
        dangerous_codes = [
            "IMPORT OS",
            "Import Sys",
            "EXEC('print(1)')",
            "Open('file.txt')",
        ]

        for dangerous_code in dangerous_codes:
            result = self.executor.execute(dangerous_code)
            assert "Error: Code contains potentially dangerous operations" in result

    # Performance and limits tests
    def test_large_computation(self):
        """Test handling of larger computations."""
        code = """
import math
result = sum(math.sqrt(i) for i in range(1, 101))
print(f"Sum of square roots 1-100: {round(result, 2)}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "Sum of square roots" in result

    def test_loop_execution(self):
        """Test execution of loops."""
        code = """
total = 0
for i in range(1, 11):
    total += i
print(f"Sum 1-10: {total}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "Sum 1-10: 55" in result

    def test_function_definition_and_call(self):
        """Test defining and calling functions."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"5! = {result}")
"""
        result = self.executor.execute(code)
        assert "Code executed successfully:" in result
        assert "5! = 120" in result


def run_tests():
    """Run all tests manually (for environments without pytest)."""
    test_instance = TestSafeCodeExecutor()
    test_methods = [
        method for method in dir(test_instance) if method.startswith("test_")
    ]

    passed = 0
    failed = 0

    print("Running SafeCodeExecutor tests...")
    print("=" * 50)

    for method_name in test_methods:
        try:
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            print(f"✅ {method_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    success = run_tests()
    exit(0 if success else 1)
