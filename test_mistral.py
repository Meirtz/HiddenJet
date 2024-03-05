from transformers import AutoModelForCausalLM, AutoTokenizer
from ModelWrapper import ModelWrapper

# Setup
device = "cuda"  # or "cpu" if CUDA is not available
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Wrap the model with ModelWrapper
wrapped_model = ModelWrapper(model)

# Prepare the input text
input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Extract hidden states from specified layers
layer_indices = [1, 2, 3]
hidden_states = wrapped_model.extract_hidden_states(inputs, layer_indices)

print("Extracted hidden states:")
print("Number of layers:", len(hidden_states))
print("Hidden state shape:", hidden_states[0].shape)  # (batch_size, sequence_length, hidden_size)

# Generate output before injection
output_before_injection = model.generate(**inputs, max_new_tokens=5)
print("\nOutput before injection:", tokenizer.decode(output_before_injection[0], skip_special_tokens=True))

# Modify the extracted hidden states
modified_hidden_states = hidden_states * 0.5

# Inject the modified hidden states back into the model
injection_layer = 2
outputs_after_injection = wrapped_model.inject_hidden_states(inputs, modified_hidden_states, injection_layer)

# Generate output after injection
output_after_injection = model.generate(**inputs, max_new_tokens=5)
print("Output after injection:", tokenizer.decode(output_after_injection[0], skip_special_tokens=True))