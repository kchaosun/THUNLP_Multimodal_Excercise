import math
from typing import List, Optional
import json
import torch
import torchvision

from threading import Thread
from copy import deepcopy
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

from .configuration import ModelConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler

from .image_processing import ModelImageProcessor
from .processing import ModelProcessor
from .llm.llm_architecture import LLMPreTrainedModel, LLMForCausalLM


class MLLMPreTrainedModel(LLMPreTrainedModel):
    config_class = ModelConfig


class MLLMModel(MLLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMForCausalLM(config)
        self.vpm = self.init_vision_module() # Vision Module
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.processor = None

        self.terminators = ['<|im_end|>', '<|endoftext|>']

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            # not suport sdpa
            self.config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_hidden_states(data)

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        ### ===>  TODO: 合并 vision_hidden_states 与 vllm_embedding，
        # 其中，vision_hidden_states 为视觉编码，当前 vllm_embedding 仅为语言模型编码
        # vision_hidden_states 为 List[torch.Tensor(num_img_in_sample, query_num, embed_dim)], query_num = 64, 与image_bound的左右边界之差相同
        # vllm_embedding 为 torch.Tensor(bsz, seq_len, embed_dim)
        
        image_bounds = data['image_bound']
        # 注意将image_bound中始末相差不是64的去掉,这些不是图片
        for i in range(len(image_bounds)):
            mask = torch.ones(len(image_bounds[i]), dtype=torch.bool)
            for j in range(len(image_bounds[i])):
                if (image_bounds[i][j][0] - image_bounds[i][j][1]).abs() != 64:
                    mask[j] = False
            image_bounds[i] = image_bounds[i][mask]

        for i in range(bs):
            cur_vision_hidden_states = vision_hidden_states[i]
            for j in range(len(cur_vision_hidden_states)):
                slice_start_idx = image_bounds[i][j][0]
                slice_end_idx = image_bounds[i][j][1]
                vllm_embedding[i][slice_start_idx:slice_end_idx] = cur_vision_hidden_states[j]
        ### <===

        return vllm_embedding, vision_hidden_states

    def get_vision_hidden_states(self, data):
        """
        data:
            input_ids: torch.Tensor(bsz, seq_len) 输入的文本id
            image_bound: List[List[Tuple[int, int]]] 图像在输入文本中的左右边界
            pixel_values: List[List[torch.Tensor(3, patch_size, H * W / patch_size)]] 输入的图像数据, 外层list是batch, 内层list是该样本中包含的分片图像张量(可能有一个或多个图像,每个图像被分成多个patch)
            具体见函数mllm/model/image_processing.py中的ModelImageProcessor类的preprocess方法,具体是先对图片做slice分片操作,再处理成多个patch size的大小
            首先理解什么是patch_size: ViT或其他基于分块的模型中的一个关键参数,表示将图像分割成的每个小方块patch的尺寸。例如patch_size = 16 表示将图像划分为 16x16 的小块(视为1个新的像素)
            tgt_sizes: List[Tensor[num_img, 2]] 存储每个图像切片在分块后的目标尺寸(即slice_image的分块网格的大小 H // patch_size, W // patch_size),外层list是batch
            vision_hidden_states: List[torch.Tensor(num_img_in_sample, query_num, embed_dim)] 预先缓存的视觉特征, 可选

        output:
            vision_hidden_states: List[torch.Tensor(num_img_in_sample, query_num, embed_dim)] num_img_in_sample是每个样本中包含的分片图像张量(可能有一个或多个图像,每个图像被分成多个patch), query_num是query的数量, embed_dim是嵌入维度
        """
        
        # 将输入数据data中的图像信息pixel values通过视觉编码器vpm编码成视觉特征hidden states
        # 再使用resampler(比如Perceiver Resampler，相当于桥连接层Adapter)将不定长的视觉token映射为LLM需要的语言token的维度
        
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']            
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values]) # flatten and permute to (H * W / patch_size, 3 * patch_size)

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                   padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L) # (B, 3, patch_size, max(H_i * W_i / patch_size))
                # print("==========get_vision_hidden_states==========")
                # print(f"all_pixel_values shape: {all_pixel_values.shape}")

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes).last_hidden_state
                
                # print(f"Batch vision_embedding shape: {vision_embedding.shape}")
                vision_embedding = self.resampler(vision_embedding, tgt_sizes) 
                # print(f"Batch vision_embedding after resampler shape: {vision_embedding.shape}")

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else: # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        return vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        
        ### ===> TODO: 实现语言模型 generate
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            **kwargs
        }
        with torch.inference_mode():
            output = self.llm.generate(**generation_kwargs)
        ### <===
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        
        ### TODO: ===> 编写输出解码过程
        # 其中应该去除tokenizer.bos_id（句子起始特殊符号），以及terminators中的符号
        result_ids = result_ids.tolist()
        print(f"result_ids: {result_ids}")

        for i in range(len(result_ids)):
            try:
                start_idx = result_ids[i].index(tokenizer.bos_id)
                result_ids[i] = result_ids[i][start_idx+1:]
            except:
                pass
            for terminator in terminators:
                try:
                    end_idx = result_ids[i].index(terminator)
                    result_ids[i] = result_ids[i][:end_idx]
                except:
                    pass

        result_text = tokenizer.batch_decode(result_ids)
        ### <===
        return result_text

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        ### ===> TODO: 实现多模态大模型的 generation，注意不要计算模型参数的梯度。
        # 1. 获取模型视觉信号
        # 2. 实现 self._decode()，返回解码后的文本
        result = None
        self.eval()
        
        with torch.inference_mode():
            model_inputs = self._to_device(model_inputs, self.device)
            vllm_embedding, vision_hidden_states = self.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(
                        inputs_embeds=vllm_embedding,
                        tokenizer=tokenizer,
                        attention_mask=attention_mask.to(self.device),
                        **kwargs
                    )

            else:
                result = self._decode(
                        inputs_embeds=vllm_embedding,
                        tokenizer=tokenizer,
                        attention_mask=attention_mask.to(self.device),
                        decode_text=decode_text,
                        **kwargs
                    )
        ### <===
        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result
    
    def _to_device(self, data, device):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
            elif isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], torch.Tensor):
                        value[i] = value[i].to(device)
                    elif isinstance(value[i], list):
                        for j in range(len(value[i])):
                            if isinstance(value[i][j], torch.Tensor):
                                value[i][j] = value[i][j].to(device)
        return data
