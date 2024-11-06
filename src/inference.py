# Test the model with some new Java code

import torch
from model import CodeTranslator

# Initialize the translator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

translator = CodeTranslator(model_name="java_to_python_translator", device=device)


test_cases = [
    """
    public int sum(int a, int b) {
        return a + b;
    }
    """,
    """
    public double average(int[] numbers) {
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        return (double) sum / numbers.length;
    }
    """,
    """
    void function(){
        int a = 5;
    }
    """,
    """
    int sum(){
        int a=6;
        int b=5;
        int sum=a+b;
        return sum;
    }
    """,
]

print("\nTesting the model with new Java code:")
for java_code in test_cases:
    print("\nInput Java code:")
    print(java_code.strip())
    print("\nTranslated Python code:")
    translated = translator.translate(java_code)
    print(translated.strip())
    print("-" * 50)
