import torch
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from contextlib import contextmanager


class ModelWrapper:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.hooks = []

    def _find_layer(self, layer_idx: int) -> torch.nn.Module:
        layers = self._get_layers()
        return layers[layer_idx]

    def _get_decoder_layers(self) -> List[torch.nn.Module]:
        return [layer for layer in self.model.model.layers]

    def _get_layers(self) -> List[torch.nn.Module]:
        def find_longest_modulelist(model, path=""):
            longest_path = path
            longest_len = 0
            for name, child in model.named_children():
                if isinstance(child, torch.nn.ModuleList) and len(child) > longest_len:
                    longest_len = len(child)
                    longest_path = f"{path}.{name}" if path else name
                child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
                if child_len > longest_len:
                    longest_len = child_len
                    longest_path = child_path
            return longest_path, longest_len

        def get_nested_attr(obj, attr_path):
            attrs = attr_path.split(".")
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj

        longest_path, _ = find_longest_modulelist(self.model)
        return get_nested_attr(self.model, longest_path)

    @contextmanager
    def _modified_forward_context(self, modifiers: Optional[List] = None):
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
        self.model.config.output_hidden_states = True

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        decoder_layers = self._get_decoder_layers()
        selected_hidden_states = [hidden_states[i + 1] for i in layer_indices]

        if return_tensors == "pt":
            return selected_hidden_states
        else:
            raise ValueError("Unsupported return_tensors value: {}".format(return_tensors))

    def inject_hidden_states(
            self,
            inputs: Dict,
            modified_hidden_states: List[torch.Tensor],
            injection_layers: List[int],
            forward_kwargs: Optional[dict] = None,
    ) -> Tuple[CausalLMOutputWithPast, List[torch.Tensor]]:
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
        decoder_layers = self._get_decoder_layers()
        for idx in injection_layers:
            layer = decoder_layers[idx]
            modifiers.append(layer.register_forward_hook(_inject_hidden_state(idx)))

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        injected_hidden_states = []

        def _collect_hidden_states(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            injected_hidden_states.append(output.detach().cpu())

        collect_modifiers = [layer.register_forward_hook(_collect_hidden_states) for layer in decoder_layers]

        with self._modified_forward_context(modifiers + collect_modifiers):
            with torch.no_grad():
                outputs = self.model(**inputs, **(forward_kwargs or {}))

        return outputs, injected_hidden_states

    def trace_forward(
            self,
            inputs: Dict,
            forward_kwargs: Optional[dict] = None,
    ) -> Tuple[CausalLMOutputWithPast, 'ForwardTrace']:
        def _trace_forward(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.forward_trace.add(module, output.cpu())

        modifiers = [layer.register_forward_hook(_trace_forward) for layer in self._get_layers()]

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        self.forward_trace = ForwardTrace()
        with self._modified_forward_context(modifiers):
            with torch.no_grad():
                outputs = self.model(**inputs, **(forward_kwargs or {}))

        return outputs, self.forward_trace

    def disable_layers(
            self,
            inputs: Dict,
            layers_to_disable: Iterable[int],
            forward_kwargs: Optional[dict] = None,
    ) -> CausalLMOutputWithPast:
        def _disable_layers():
            original_layers = self._get_layers()
            new_layers = torch.nn.ModuleList(
                [layer for i, layer in enumerate(original_layers) if i not in layers_to_disable]
            )
            self._set_layers(new_layers)
            yield
            self._set_layers(original_layers)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self._modified_forward_context([_disable_layers()]):
            with torch.no_grad():
                outputs = self.model(**inputs, **(forward_kwargs or {}))

        return outputs

    def generate(
            self,
            inputs: Dict,
            generate_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        input_ids = self.model.generate(
            **inputs,
            **(generate_kwargs or {}),
        )

        return input_ids

    def injected_generate(
            self,
            inputs: Dict,
            modified_hidden_states: List[torch.Tensor],
            injection_layers: List[int],
            generate_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
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
        decoder_layers = self._get_decoder_layers()
        for idx in injection_layers:
            layer = decoder_layers[idx]
            modifiers.append(layer.register_forward_hook(_inject_hidden_state(idx)))

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self._modified_forward_context(modifiers):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **(generate_kwargs or {}),
                )

        return output_ids


class ForwardTrace:
    def __init__(self):
        self.traces = []

    def add(self, module, output):
        self.traces.append((module, output))

    def __getitem__(self, idx):
        return self.traces[idx]

    def __len__(self):
        return len(self.traces)