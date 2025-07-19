import io
import os
import copy
import json
import base64
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from mllm.model import MLLMModel
from mllm.model.processing import ModelProcessor
from mllm.model.image_processing import ModelImageProcessor
from utils.file_io import read_jsonlines, read_json

class MLLMEvalModel(MLLMModel):
    def __init__(self, config):
        super().__init__(config)

    def chat(
        self,
        image,
        msgs,
        tokenizer,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192,
        system_prompt='',
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        assert self.config.query_num == processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.patch_size == processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.use_image_id == processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_mode == processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        assert sampling or not stream, "if use stream mode, make sure sampling=True"

        prompts_lists, input_images_lists = self.prepare_chat_inputs(tokenizer, system_prompt, msgs_list, images_list)
        
        print("====1====")
        print(f"prompts_lists: {prompts_lists}")

        inputs = processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            return_tensors="pt",
            max_length=max_inp_length
        ).to(self.device)
        print(f"inputs_ids: {inputs['input_ids'].size()}")
        print(f"attention_mask: {inputs['attention_mask'].size()}")
        print(f"pixel_values shape :({len(inputs['pixel_values'])}, {len(inputs['pixel_values'][0])})")
        print(f"tensor in pixel_values shape: {inputs['pixel_values'][0][0].size()}")
        print(f"image_sizes: {inputs['image_sizes']}")
        print(f"image_bound: {inputs['image_bound']}")
        print(f"tgt_sizes: {inputs['tgt_sizes']}")

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config['min_new_tokens'] = min_new_tokens

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                decode_text=True,
                **generation_config
            )

        if stream:
            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, '')
                    yield text
            return stream_gen()

        else:
            if batched:
                answer = res
            else:
                answer = res[0]
            return answer

    def prepare_chat_inputs(self, tokenizer, system_prompt, msgs_list, images_list):
        ### ===> TODO:
        # 将输入文本转换为预处理函数所需的格式
        # Rule:
        # 1. 输入图片的位置应该替换为 (<image>./</image>) 字符串
        # 2. 使用 tokenizer 将输入文本转换为模型所需的输入格式，并进行分词（tokenize）
        # 提示：使用 tokenizer.apply_chat_template 进行输入文本格式转换

        prompts_lists = []
        input_images_lists = []

        system_msg = [{
            "role": "system",
            "content": system_prompt if system_prompt else "You are a helpful assistant."
        }]
        
        print(f"msgs_list: {msgs_list}")

        for i in range(len(msgs_list)):
            input_images_lists.append(images_list[i])
            
            user_msg = msgs_list[i][0]
            
            if user_msg['role'] == 'user':
                content = user_msg['content']
                content = f'<image>./</image>{content}'
                user_msg['content'] = content

            msg = system_msg + [user_msg]

            prompt = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts_lists.append(prompt)
            print(f"Processed prompt {i}: {prompt}")

        ### <===

        return prompts_lists, input_images_lists


def eval_model(args):
    model = MLLMEvalModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    img_processor_config = read_json('mllm/model/mllm_preprocessor_config.json')
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)

    model.eval().cuda()

    input_data = read_jsonlines(args.question_file)
    print(f"Loaded {len(input_data)} items from {args.question_file}")

    ans_file = open(args.answers_file, 'w')

    with torch.inference_mode():
        i = 0
        for item in tqdm(input_data):
            image = item['image']
            msgs = [{"role": "user", "content": item['question']}]

            if len(image) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
            
            print(f"Processing item {i} with question: {item['question']}")

            answer = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor
            )
            print(f"Generated answer for item {i}: {answer}")

            answer_dict = {
                "idx": i,
                "question": msgs[0]['content'],
                "answer": answer,
                "model": args.model_name_or_path,
                "metainfos": {key: value for key, value in item.items() if key not in ['image', 'question']}
            }

            if 'image_id' in item.keys():
                answer_dict['image_id'] = item['image_id']

            ans_file.write(
                json.dumps(answer_dict) + '\n'
            )
            ans_file.flush()

            i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--sampling", action='store_true')
    args = parser.parse_args()

    eval_model(args)