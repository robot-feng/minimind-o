import argparse
import json
import os
import random
import time
import warnings
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, MimiModel
from model.model_omni import MiniMindOmni, OmniConfig
from dataset.omni_dataset import OmniDataset
from dataset.video import DEFAULT_VIDEO_FRAMES, VIDEO_EXTENSIONS, prepare_image_inputs, prepare_video_inputs
from trainer.audio_output import save_generated_audio
from trainer.trainer_utils import setup_seed, log_model_params
warnings.filterwarnings('ignore')


def save_visual_result(path, mode, source, prompt, answer):
    if not path or answer is None:
        return
    result = {"mode": mode, "source": source, "prompt": prompt, "answer": answer}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindOmni(
            OmniConfig(
                hidden_size=args.hidden_size, 
                num_hidden_layers=args.num_hidden_layers, 
                use_moe=bool(args.use_moe)
            ),
            audio_encoder_path=None if args.text_only else "./model/SenseVoiceSmall",
            vision_model_path=args.vision_dir
        )
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        if args.text_only:
            model.audio_encoder, model.audio_processor = None, None
        else:
            model.audio_encoder, model.audio_processor = MiniMindOmni.load_sensevoice("./model/SenseVoiceSmall")
        model.vision_encoder, model.vision_processor = MiniMindOmni.load_vision(args.vision_dir)
    log_model_params(model)
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    model.mimi_model = None if args.text_only else MimiModel.from_pretrained("./model/mimi").eval()
    return model.half().eval().to(args.device), tokenizer


def eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, output_name, pixel_values=None, history=None, audio_lens=None, ref_codes=None, spk_emb=None):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
    x = torch.tensor(tokenizer(inputs_text).data['input_ids'], dtype=torch.long, device=args.device)[None, ...]

    if args.text_only:
        output_ids = model.generate_text(
            x, eos_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, pixel_values=pixel_values,
        )
        answer = tokenizer.decode(output_ids[0, x.size(1):].tolist(), skip_special_tokens=True)
        print('📒 [Thinker]: ', answer, flush=True)
        return answer

    audio_frames = []
    with torch.no_grad():
        res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p, stream=True,
                               return_audio_codes=True, open_thinking=bool(args.open_thinking),
                               audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values,
                               ref_codes=ref_codes, spk_emb=spk_emb)
        print('📒 [Thinker]: ', end='', flush=True)
        history_idx = 0
        for y, audio_frame in res_y:
            if y is not None:
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                if answer and answer[-1] != '�':
                    print(answer[history_idx:], end='', flush=True)
                    history_idx = len(answer)
            if audio_frame:
                audio_frames.append(audio_frame)
        print()

        if audio_frames:
            print(f'🎹 [Talker]: {len(audio_frames)} frames', end=" ")
            if args.decode_audio:
                try:
                    codes = [f for f in audio_frames if f and len(f) == 8]
                    if not codes:
                        print('⚠️  生成的Mimi codes为空，跳过保存。')
                        return
                    mimi_codes = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0).to(args.device)
                    filtered = torch.where(mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes)
                    audio = model.mimi_model.decode(filtered).audio_values
                    output_path = os.path.join(args.output_dir, output_name)
                    saved_path, error = save_generated_audio(
                        audio.squeeze().float().cpu().numpy(), output_path
                    )
                    if error:
                        print(f'| MP3 export failed ({error}); WAV saved to: {saved_path}')
                    else:
                        print(f'| Audio decoded to: {saved_path}')
                except Exception as e:
                    print(f'⚠️  保存音频失败: {str(e)}')
            else:
                print("(decode_audio=off)\n")


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O Chat")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_omni', type=str, help="权重名称前缀")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="Thinker生成温度")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值")
    parser.add_argument('--output_dir', default='./output_audio/', type=str, help="输出音频保存目录")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--audio_dir', default='./dataset/eval_omni/', type=str, help="测试音频目录")
    parser.add_argument('--image_dir', default='./dataset/eval_omni/', type=str, help="测试图像目录")
    parser.add_argument('--video_dir', default='./dataset/eval_omni/', type=str, help="测试视频目录")
    parser.add_argument('--video_frames', default=DEFAULT_VIDEO_FRAMES, type=int, help="统一视觉帧数；单图复制到该帧数")
    parser.add_argument('--vision_dir', default='google/tipsv2-b14', type=str, help="TIPSv2视觉模型 ID 或本地路径")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启思考模式（0=否，1=是）（思考模式下禁用audio输出）")
    parser.add_argument('--text_only', action='store_true', help="仅生成文本，不加载或运行音频模块；适用于文本、图像和视频评估")
    parser.add_argument('--results_jsonl', type=str, help="仅文本图像/视频评测：逐样本保存来源、提示和回答")
    parser.add_argument('--seed', type=int, help="固定随机种子以便复现实验")
    parser.add_argument('--decode_audio', default=1, type=int, help="是否解码音频输出（0=否，1=是）")
    parser.add_argument('--mode', default='0', type=str, help="评估模式：-1=all 0=text 1=multi 2=audio 3=clone 4=image 5=mix 6=video（逗号组合，如 2,5）")
    parser.add_argument('--prompt_lang', default=0, type=int, choices=[0, 1, 2], help="问题语言：0=英文 1=中文 2=英文+中文")
    args = parser.parse_args()
    modes = set(args.mode.replace(',', '').replace('-1', '0123456'))
    if args.text_only and modes.intersection({'2', '3', '5'}):
        parser.error("--text_only cannot be combined with audio or mixed-input modes 2, 3, or 5")
    if args.results_jsonl and (not args.text_only or not modes.intersection({'4', '6'})):
        parser.error("--results_jsonl requires --text_only and image or video mode 4/6")
    
    if not args.text_only:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.results_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_jsonl)), exist_ok=True)
        with open(args.results_jsonl, 'w', encoding='utf-8'):
            pass
    model, tokenizer = init_model(args)
    setup_seed(args.seed if args.seed is not None else int(time.time()) % 31415926)

    if '0' in modes:
        print('\n\n==================== text -> {text, audio} ====================')
        test_prompts_en = [
            "Tell me an interesting fact about space.", "How do I make a cup of coffee?", "What's the weather like today?",
            "Will it rain tomorrow?", "Tell me a joke.", "Can you sing a song for me?", "Please introduce yourself."
        ]
        test_prompts_zh = [
            "告诉我一个关于太空的有趣事实。", "如何制作一杯咖啡？", "今天的天气怎么样？",
            "明天会下雨吗？", "给我讲个笑话吧", "你能为我唱首歌吗？", "介绍一下你自己"
        ]
        test_prompts = [test_prompts_en, test_prompts_zh, test_prompts_en + test_prompts_zh][args.prompt_lang]
        for idx, prompt in enumerate(test_prompts):
            print(f'\n📝 [text-{idx+1}]: {prompt}')
            eval_sample(model, tokenizer, args, idx, prompt, None, f"text-{idx:02d}.mp3")

    if '1' in modes:
        print('\n\n==================== multi-turn -> {text, audio} ====================')
        multi_turn_tests_zh = [
            {
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
                ],
                "prompt": "我想找点事做，你有什么建议吗？"
            },
            {
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
                    {"role": "user", "content": "我想找点事做，你有什么建议吗？"},
                    {"role": "assistant", "content": "可以听听音乐或者看看书，放松一下心情。"}
                ],
                "prompt": "好的，那我去照做了，谢谢你"
            }
        ]
        multi_turn_tests_en = [
            {
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"}
                ],
                "prompt": "I want to find something to do. Do you have any suggestions?"
            },
            {
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                    {"role": "user", "content": "I want to find something to do. Do you have any suggestions?"},
                    {"role": "assistant", "content": "You can listen to music or read a book to relax a little."}
                ],
                "prompt": "Okay, I will try that. Thank you."
            }
        ]
        multi_turn_tests = [multi_turn_tests_en, multi_turn_tests_zh, multi_turn_tests_en + multi_turn_tests_zh][args.prompt_lang]
        for idx, test in enumerate(multi_turn_tests):
            print(f'\n💬 [multi-{idx+1}]')
            for msg in test["history"]: print(f'   {msg["role"]}: {msg["content"]}')
            print(f'   user: {test["prompt"]}')
            eval_sample(model, tokenizer, args, idx, test["prompt"], None, f"multi-{idx:02d}.mp3", history=test["history"])

    if '2' in modes:
        print('\n\n==================== audio -> {text, audio} ====================')
        audio_files_en = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('audio-en-') and f.lower().endswith(('.mp3', '.wav'))])
        audio_files_zh = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('audio-zh-') and f.lower().endswith(('.mp3', '.wav'))])
        audio_files = [audio_files_en, audio_files_zh, audio_files_en + audio_files_zh][args.prompt_lang]
        for idx, audio_file in enumerate(audio_files):
            print(f'\n🎤 [audio-{idx+1}]: {audio_file}')
            mel, valid_len = OmniDataset.process_audio(os.path.join(args.audio_dir, audio_file), model.audio_processor)
            audio_inputs = mel.unsqueeze(0).to(args.device)
            audio_lens = torch.tensor([valid_len], device=args.device)
            audio_token_len = valid_len or 1
            prompt = model.config.audio_special_token * audio_token_len
            eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"audio-{idx:02d}-{os.path.splitext(audio_file)[0]}.mp3", audio_lens=audio_lens)

    if '3' in modes:
        print('\n\n==================== clone voice -> {text, audio} ====================')
        clone_prompts_en = ["Hello, please introduce yourself.", "What's the weather like today?", "Tell me a joke."]
        clone_prompts_zh = ["你好，请介绍一下你自己。", "今天天气怎么样？", "给我讲个笑话吧。"]
        clone_prompts = [clone_prompts_en, clone_prompts_zh, clone_prompts_en + clone_prompts_zh][args.prompt_lang]
        voices_pt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'speaker', 'voices_unseen.pt')
        voices = [('default', None, None)]
        if os.path.exists(voices_pt):
            voice_data = torch.load(voices_pt, map_location=args.device)
            for speaker, v in sorted(voice_data.items()):
                rc = v['ref_codes'].unsqueeze(0).to(args.device)
                se = v['spk_emb'].half().unsqueeze(0).to(args.device) if 'spk_emb' in v else None
                voices.append((speaker, rc, se))
        for speaker, rc, se in voices:
            info = f'ref_codes: {rc.shape[2]} frames, spk_emb: {"+" if se is not None else "-"}' if rc is not None else ('spk_emb only' if se is not None else 'default')
            print(f'\n🎵 [clone: {speaker}] {info}')
            for idx, prompt in enumerate(clone_prompts):
                print(f'  📝 [text-{idx+1}]: {prompt}')
                history = [{"role": "system", "content": "你是一个专业的语音助手，请用给定的音色风格来回答用户的问题。请尽量详细地回答，给出有价值的信息。"}]
                eval_sample(model, tokenizer, args, idx, prompt, None, f"clone-{speaker}-{idx:02d}.mp3", ref_codes=rc, history=history, spk_emb=se)

    if '4' in modes:
        print('\n\n==================== image -> {text, audio} ====================')
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for idx, image_file in enumerate(image_files):
            print(f'\n🖼️ [image-{idx+1}]: {image_file}')
            image = Image.open(os.path.join(args.image_dir, image_file)).convert('RGB')
            pixel_values, image_frame_prompt = prepare_image_inputs(
                image, model.vision_processor, model.config, args.device, args.video_frames
            )
            prompts = [["Please describe this image."], ["请描述这张图片"], ["Please describe this image.", "请描述这张图片"]][args.prompt_lang]
            for lang_idx, prompt_text in enumerate(prompts):
                prompt = prompt_text + "\n\n" + image_frame_prompt
                answer = eval_sample(model, tokenizer, args, idx, prompt, None,
                                     f"image-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3",
                                     pixel_values=pixel_values)
                save_visual_result(args.results_jsonl, "image", image_file, prompt_text, answer)

    if '5' in modes:
        print('\n\n==================== text+audio+image -> {text, audio} ====================')
        img_audio_files = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('img-') and f.lower().endswith(('.mp3', '.wav'))])
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        text_hints = [["Please answer me: "], ["请回答我："], ["Please answer me: ", "请回答我："]][args.prompt_lang]
        for idx, image_file in enumerate(image_files):
            audio_file = random.choice(img_audio_files)
            image = Image.open(os.path.join(args.image_dir, image_file)).convert('RGB')
            pixel_values, image_frame_prompt = prepare_image_inputs(
                image, model.vision_processor, model.config, args.device, args.video_frames
            )
            for lang_idx, text_hint in enumerate(text_hints):
                print(f'\n🌀 [mix-{idx+1}-{lang_idx}]: {text_hint} | {audio_file} | {image_file}')
                mel, valid_len = OmniDataset.process_audio(os.path.join(args.audio_dir, audio_file), model.audio_processor)
                audio_inputs = mel.unsqueeze(0).to(args.device)
                audio_lens = torch.tensor([valid_len], device=args.device)
                audio_token_len = valid_len or 1
                prompt = text_hint + model.config.audio_special_token * audio_token_len + "\n\n" + image_frame_prompt
                eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"mix-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3", pixel_values=pixel_values, audio_lens=audio_lens)

    if '6' in modes:
        print('\n\n==================== video -> {text, audio} ====================')
        video_files = sorted(f for f in os.listdir(args.video_dir)
                             if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)
        prompts = [["Describe this video."], ["请描述这个视频"], ["Describe this video.", "请描述这个视频"]][args.prompt_lang]
        for idx, video_file in enumerate(video_files):
            video_path = os.path.join(args.video_dir, video_file)
            pixel_values, frame_prompt = prepare_video_inputs(
                video_path, model.vision_processor, model.config, args.device, args.video_frames
            )
            for lang_idx, prompt_text in enumerate(prompts):
                prompt = f"{prompt_text}\n\n{frame_prompt}"
                answer = eval_sample(model, tokenizer, args, idx, prompt, None,
                                     f"video-{idx:02d}-{lang_idx}-{os.path.splitext(video_file)[0]}.mp3",
                                     pixel_values=pixel_values)
                save_visual_result(args.results_jsonl, "video", video_file, prompt, answer)


if __name__ == "__main__":
    main()
