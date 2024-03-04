from transformers import AutoModelForCausalLM, AutoTokenizer
# Assuming ModelWrapper.py is in the same directory or in the Python path
from ModelWrapper import ModelWrapper

device = "cuda"  # the device to load the model onto

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Wrap the model with ModelWrapper
wrapped_model = ModelWrapper(model)

# Prepare the input text
input_text = "The capital of France is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Extract hidden states from the last layer
hidden_states = wrapped_model.extract_hidden_states(input_ids, layer_indices=[-1])

# Display the shape of the hidden states tensor
# Should be [batch_size, sequence_length, hidden_size] for the last layer
print(hidden_states[0].shape)
