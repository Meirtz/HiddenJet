from transformers import AutoModelForCausalLM, AutoTokenizer
# Assuming ModelWrapper.py is in the same directory or in the Python path
from ModelWrapper import ModelWrapper
# Assuming ModelWrapper.py is correctly imported along with other necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup
device = "cuda"  # or "cpu" if CUDA is not available
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_8bit=True, device_map="auto")#.to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Wrap the model with ModelWrapper
wrapped_model = ModelWrapper(model)

# Prepare the input text properly
input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)  # Ensure to move to the appropriate device

# Extract hidden states from the last layer
hidden_states = wrapped_model.extract_hidden_states(inputs, layer_indices=[1,2,3])

print(len(hidden_states), hidden_states[0].shape)  # (1, sequence_length, hidden_size)
print(hidden_states[0].shape)  # (batch_size, sequence_length, hidden_size)
