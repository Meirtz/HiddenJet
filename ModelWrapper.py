import torch
import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []

    def _find_layer(self, layer_idx: int) -> torch.nn.Module:
        """
        Find the layer at the specified index in the model's decoder layers.

        Args:
            layer_idx (int): The index of the layer to find.

        Returns:
            torch.nn.Module: The layer at the specified index.
        """
        layers = self._get_decoder_layers()
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        return layers[layer_idx]

    def _get_decoder_layers(self) -> List[torch.nn.Module]:
        """
        Get the decoder layers of the model.

        Returns:
            List[torch.nn.Module]: The decoder layers of the model.
        """
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise AttributeError("Model does not have the expected structure")
        return [layer for layer in self.model.model.layers]

    @contextmanager
    def _modified_forward_context(self, modifiers: Optional[List] = None):
        """
        Context manager for modifying the forward pass of the model.

        Args:
            modifiers (Optional[List]): A list of modifiers to apply during the forward pass.
        """
        if modifiers is None:
            modifiers = []
        try:
            for modifier in modifiers:
                modifier.__enter__()
            yield
        finally:
            for modifier in modifiers:
                modifier.__exit__(None, None, None)

    def extract_hidden_states(
            self,
            inputs: Dict,
            layer_indices: List[int],
            return_tensors: str = "pt",
    ) -> List[torch.Tensor]:
        """
        Extract the hidden states from the specified layers of the model.

        Args:
            inputs (Dict): The input data for the model.
            layer_indices (List[int]): The indices of the layers to extract hidden states from.
            return_tensors (str): The format of the returned tensors. Defaults to "pt".

        Returns:
            List[torch.Tensor]: The extracted hidden states from the specified layers.
        """
        self.model.config.output_hidden_states = True

        inputs = {k: v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        selected_hidden_states = [hidden_states[i + 1] for i in layer_indices]

        if return_tensors == "pt":
            return selected_hidden_states
        else:
            raise ValueError(f"Unsupported return_tensors value: {return_tensors}")

    def inject_hidden_states(
            self,
            inputs: Dict,
            modified_hidden_states: List[torch.Tensor],
            injection_layers: List[int],
            forward_kwargs: Optional[dict] = None,
    ) -> Tuple[CausalLMOutputWithPast, List[torch.Tensor]]:
        """
        Inject modified hidden states into the specified layers of the model and return the output and injected hidden states.

        Args:
            inputs (Dict): The input data for the model.
            modified_hidden_states (List[torch.Tensor]): The modified hidden states to inject.
            injection_layers (List[int]): The indices of the layers to inject the modified hidden states into.
            forward_kwargs (Optional[dict]): Additional keyword arguments for the forward pass. Defaults to None.

        Returns:
            Tuple[CausalLMOutputWithPast, List[torch.Tensor]]: A tuple containing the model's output and the injected hidden states.
        """

        def _inject_hidden_state(layer_idx):
            def _injection_hook(module, inputs, output):
                if layer_idx in injection_layers:
                    hidden_state_idx = injection_layers.index(layer_idx)
                    with torch.no_grad():
                        if isinstance(output, tuple):
                            hidden_state = output[0]
                            hidden_state.add_(modified_hidden_states[hidden_state_idx])
                            output = (hidden_state,) + output[1:]
                        else:
                            output.add_(modified_hidden_states[hidden_state_idx])
                return output

            return _injection_hook

        modifiers = []
        for idx in injection_layers:
            layer = self._find_layer(idx)
            modifiers.append(layer.register_forward_hook(_inject_hidden_state(idx)))

        inputs = {k: v for k, v in inputs.items()}

        injected_hidden_states = []

        def _collect_hidden_states(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            injected_hidden_states.append(output.detach().cpu())

        collect_modifiers = [layer.register_forward_hook(_collect_hidden_states) for layer in
                             self._get_decoder_layers()]

        with self._modified_forward_context(modifiers + collect_modifiers):
            with torch.no_grad():
                outputs = self.model(**inputs, **(forward_kwargs or {}))

        return outputs, injected_hidden_states

    def injected_generate(
            self,
            inputs: Dict,
            modified_hidden_states: List[torch.Tensor],
            injection_layers: List[int],
            generate_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Generate text using the model with injected modified hidden states.

        Args:
            inputs (Dict): The input data for the model.
            modified_hidden_states (List[torch.Tensor]): The modified hidden states to inject.
            injection_layers (List[int]): The indices of the layers to inject the modified hidden states into.
            generate_kwargs (Optional[dict]): Additional keyword arguments for the generate method. Defaults to None.

        Returns:
            torch.Tensor: The generated text.
        """

        def _inject_hidden_state(layer_idx):
            def _injection_hook(module, inputs, output):
                if layer_idx in injection_layers:
                    hidden_state_idx = injection_layers.index(layer_idx)
                    with torch.no_grad():
                        if isinstance(output, tuple):
                            hidden_state = output[0]
                            modified_hidden_state = modified_hidden_states[hidden_state_idx]
                            seq_len = hidden_state.shape[1]
                            modified_hidden_state = modified_hidden_state[:, -seq_len:, :]
                            hidden_state.add_(modified_hidden_state)
                            output = (hidden_state,) + output[1:]
                        else:
                            modified_hidden_state = modified_hidden_states[hidden_state_idx]
                            seq_len = output.shape[1]
                            modified_hidden_state = modified_hidden_state[:, -seq_len:, :]
                            output.add_(modified_hidden_state)
                return output

            return _injection_hook

        modifiers = []
        for idx in injection_layers:
            layer = self._find_layer(idx)
            modifiers.append(layer.register_forward_hook(_inject_hidden_state(idx)))

        inputs = {k: v for k, v in inputs.items()}

        with self._modified_forward_context(modifiers):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **(generate_kwargs or {}),
                )

        return output_ids

    def dynamic_generate(
            self,
            inputs: Dict,
            extraction_layers: List[int],
            injection_layers: List[int],
            generate_kwargs: Optional[dict] = None,
    ) -> str:
        """
        Generate text using the model with dynamic extraction and injection of hidden states.

        Args:
            inputs (Dict): The input data for the model.
            extraction_layers (List[int]): The indices of the layers to extract hidden states from.
            injection_layers (List[int]): The indices of the layers to inject the modified hidden states into.
            generate_kwargs (Optional[dict]): Additional keyword arguments for the generate method. Defaults to None.

        Returns:
            str: The generated text.
        """
        generated_tokens = []
        current_inputs = inputs

        while True:
            # Extract hidden states from the specified layers
            hidden_states = self.extract_hidden_states(current_inputs, extraction_layers)

            # Modify the extracted hidden states
            modified_hidden_states = [hs * 1.5 + torch.randn_like(hs) * 0.1 for hs in hidden_states]

            # Inject the modified hidden states into the specified layers
            outputs, _ = self.inject_hidden_states(current_inputs, modified_hidden_states, injection_layers)

            # Generate the next token using the modified model
            next_token = self.model.generate(
                **current_inputs,
                **(generate_kwargs or {}),
                max_new_tokens=1,
                # do_sample=True,
                # temperature=0.7,
            )

            # Update the input data with the generated token
            current_inputs = {
                "input_ids": next_token,
                "attention_mask": torch.ones_like(next_token),
            }

            # Append the generated token to the list of generated tokens
            generated_tokens.append(next_token[0, -1].item())

            # Check if the maximum length is reached or if a stopping condition is met
            if len(generated_tokens) >= generate_kwargs.get("max_length", 100) or next_token[
                0, -1].item() == self.model.config.eos_token_id:
                break

        # Decode the generated tokens into text
        generated_text = self.tokenizer.decode(generated_tokens)

        return generated_text