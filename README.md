# Code Analysis

## Language

This code appears to be written in JavaScript.

## Functions

- `add`
- `subtract`
- `multiply`
- `divide`
- `calculateExpression`

## How to Run

1. Ensure you have JavaScript installed on your system.
2. Save the code in a file with an appropriate extension (e.g., `.javascript`).
3. Run the file using the appropriate command for JavaScript.

## Code

```javascript
// Simple calculator functions

function add(a, b) {
  return a + b;
}

function subtract(a, b) {
  return a - b;
}

function multiply(a, b) {
  return a * b;
}

function divide(a, b) {
  if (b === 0) {
    throw new Error("Cannot divide by zero");
  }
  return a / b;
}

// Example usage
const num1 = 10;
const num2 = 5;

console.log(`Addition: ${add(num1, num2)}`);
console.log(`Subtraction: ${subtract(num1, num2)}`);
console.log(`Multiplication: ${multiply(num1, num2)}`);
console.log(`Division: ${divide(num1, num2)}`);

// Advanced function
function calculateExpression(expression) {
  return new Function('return ' + expression)();
}

// Test the advanced function
console.log(`Complex calculation: ${calculateExpression('3 * (4 + 2) - 7')}`);
```
