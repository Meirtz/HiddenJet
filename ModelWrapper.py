from transformers import PreTrainedModel
import torch


class ModelWrapper:
    def __init__(self, model: PreTrainedModel):
        """
        Initialize the wrapper with a Hugging Face PreTrainedModel.

        Parameters:
        - model: A Hugging Face PreTrainedModel instance.
        """
        self.model = model
        self.model.eval()  # Ensure the model is in evaluation mode
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

    def extract_hidden_states(self, inputs, layer_indices, return_tensors="pt"):
        """
        Extract the hidden states from specified layers.

        Parameters:
        - inputs: The inputs to the model, properly formatted for the model.
        - layer_indices: A list of integers specifying which layers' hidden states to return.
        - return_tensors: The format of the returned hidden states. Defaults to PyTorch tensors ('pt').

        Returns:
        - A list of hidden states from the specified layers.
        """
        # Ensure model outputs hidden states
        self.model.config.output_hidden_states = True

        # Move inputs to the same device as the model
        inputs = {k: v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract hidden states
        hidden_states = outputs.hidden_states

        # Select the requested layers
        selected_hidden_states = [hidden_states[i] for i in layer_indices]

        if return_tensors == "pt":
            return selected_hidden_states
        else:
            raise ValueError("Unsupported return_tensors value: {}".format(return_tensors))

    def inject_hidden_states(self, inputs, hidden_states, layer_indices):
        """
        Inject custom hidden states into specified layers. This is a placeholder
        and would need to be implemented based on specific model architecture.

        Parameters:
        - inputs: The inputs to the model.
        - hidden_states: The hidden states to inject.
        - layer_indices: The layers into which to inject the hidden states.

        Returns:
        - Model outputs after injection.
        """
        raise NotImplementedError("This method requires a custom implementation.")

