from transformers import AutoModelForCausalLM, AutoTokenizer
from ModelWrapper import ModelWrapper

# Setup
device = "cuda"  # or "cpu" if CUDA is not available
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

print(model)

# Wrap the model with ModelWrapper
wrapped_model = ModelWrapper(model)

# Prepare the input text
input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Extract hidden states from specified layers
layer_indices = [x for x in range(3)]
hidden_states = wrapped_model.extract_hidden_states(inputs, layer_indices)
hidden_states_before_injection = hidden_states


print("Extracted hidden states:")
print("Number of layers:", len(hidden_states))
print("Hidden state shape:", hidden_states[0].shape)  # (batch_size, sequence_length, hidden_size)

# Generate output before injection
output_before_injection = model.generate(**inputs, max_new_tokens=5)
print("\nOutput before injection:", tokenizer.decode(output_before_injection[0], skip_special_tokens=True))

# Modify the extracted hidden states
modified_hidden_states = [hs * 0.5 for hs in hidden_states]
hidden_states_after_injection = modified_hidden_states

# Inject the modified hidden states back into the model
injection_layers = [x for x in range(3)]
outputs_after_injection = wrapped_model.inject_hidden_states(inputs, modified_hidden_states, injection_layers)

# Generate output after injection using the injected_generate method
output_after_injection = wrapped_model.injected_generate(inputs, modified_hidden_states, injection_layers, generate_kwargs={"max_new_tokens": 5})
print("Output after injection (using injected_generate):", tokenizer.decode(output_after_injection[0], skip_special_tokens=True))
print("Hidden states distance:", sum([((hs_before - hs_after) ** 2).sum() for hs_before, hs_after in zip(hidden_states_before_injection, hidden_states_after_injection)]) ** 0.5)

outputs_after_injection, injected_hidden_states = wrapped_model.inject_hidden_states(inputs, modified_hidden_states, injection_layers)

print("Injected hidden states:")
print("Number of layers:", len(injected_hidden_states))
print("Hidden state shape:", injected_hidden_states[0].shape)