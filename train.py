# Initialize the translator
import torch
from model import CodeTranslator, get_training_data


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

translator = CodeTranslator(device=device)

# Get training data
train_java, train_python = get_training_data()

# Split into train/val (75/25 split)
split = int(len(train_java) * 0.75)
val_java = train_java[split:]
val_python = train_python[split:]
train_java = train_java[:split]
train_python = train_python[:split]

# Train the model
print("Starting training...")
translator.train(
    train_java=train_java,
    train_python=train_python,
    val_java=val_java,
    val_python=val_python,
    batch_size=2,
    epochs=8,
    learning_rate=0.001,
)

# Test the model with some new Java code
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
]

print("\nTesting the model with new Java code:")
for java_code in test_cases:
    print("\nInput Java code:")
    print(java_code.strip())
    print("\nTranslated Python code:")
    translated = translator.translate(java_code)
    print(translated.strip())
    print("-" * 50)

# Save the model
translator.model.save_pretrained("java_to_python_translator_8Epochs_lr_0.001")
translator.tokenizer.save_pretrained("java_to_python_translator_8Epochs_lr_0.001")
print("\nModel saved to 'java_to_python_translator' directory")
