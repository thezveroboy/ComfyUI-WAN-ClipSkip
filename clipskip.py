import torch
import comfy.model_management as mm
import comfy.sd

class CLIPSkip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model (e.g., from CLIPLoader with type 'wan', like umt5_xxl)"}),
                "skip_layers": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 24,  # Поддержка 24 слоёв в UMT5 XXL
                    "step": 1,
                    "tooltip": "Number of CLIP layers to skip (0 = no skip)"
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "modify_clip"
    CATEGORY = "conditioning"
    DESCRIPTION = "Modifies a WAN CLIP model (e.g., umt5_xxl) to skip a specified number of encoder layers."

    def modify_clip(self, clip, skip_layers):
        if skip_layers < 0 or skip_layers > 24:
            raise ValueError("skip_layers must be between 0 and 24")

        # Клонируем объект CLIP
        modified_clip = clip.clone()
        device = mm.get_torch_device()

        # Если skip_layers = 0, возвращаем без изменений
        if skip_layers == 0:
            return (modified_clip,)

        # Получаем модель кондиционирования (WanTEModel)
        model = modified_clip.cond_stage_model
        model.eval()

        # Проверяем структуру
        if not hasattr(model, "umt5xxl") or not hasattr(model.umt5xxl, "transformer"):
            raise AttributeError("Unsupported WAN CLIP structure: expected 'umt5xxl.transformer'")
        transformer = model.umt5xxl.transformer
        
        if not hasattr(transformer, "encoder") or not hasattr(transformer.encoder, "block"):
            raise AttributeError("Unsupported WAN CLIP structure: expected 'transformer.encoder.block'")
        if not hasattr(transformer, "shared"):
            raise AttributeError("Unsupported WAN CLIP structure: expected 'transformer.shared' for embeddings")
        if not hasattr(transformer.encoder, "final_layer_norm"):
            raise AttributeError("Unsupported WAN CLIP structure: expected 'transformer.encoder.final_layer_norm'")

        # Получаем слои энкодера
        layers = transformer.encoder.block
        total_layers = len(layers)
        print(f"Total layers: {total_layers}")
        if skip_layers >= total_layers:
            raise ValueError(f"skip_layers ({skip_layers}) exceeds total layers ({total_layers})")

        # Оригинальный метод forward
        original_forward = model.forward

        def custom_forward(input_ids, attention_mask=None, **kwargs):
            with torch.no_grad():
                # Получаем входные эмбеддинги
                hidden_states = transformer.shared(input_ids)

                # Применяем только нужное количество слоёв
                target_layers = total_layers - skip_layers
                for i in range(target_layers):
                    hidden_states = layers[i](hidden_states, attention_mask=attention_mask)

                # Применяем финальную нормализацию
                final_output = transformer.encoder.final_layer_norm(hidden_states)

                # Применяем проекцию, если она есть
                if hasattr(model, "text_projection"):
                    final_output = final_output @ model.text_projection
                    print("Applied 'text_projection'")

                # Возвращаем в формате словаря
                return {"last_hidden_state": final_output}

        # Заменяем forward
        model.forward = custom_forward

        # Возвращаем модифицированный CLIP
        return (modified_clip,)

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "CLIPSkip": CLIPSkip
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSkip": "CLIP Skip (WAN)"
}
