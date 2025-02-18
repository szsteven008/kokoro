import argparse
import os
import torch
import onnx
import onnxruntime as ort
import sounddevice as sd

from kokoro import KModel, KPipeline

class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return self.model.device

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        style: torch.FloatTensor, 
        speed: torch.IntTensor
    ):
        audio = self.model.inference(input_ids, style, speed)
        return audio

def export_onnx(model, output):
    onnx_file = output + "/" + "kokoro.onnx"

    input_ids = torch.randint(1, 100, (48,)).numpy()
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.randint(1, 10, (1,)).int()

    torch.onnx.export(
        model, 
        args = (input_ids, style, speed), 
        f = onnx_file, 
        export_params = True, 
        verbose = True, 
        input_names = [ 'input_ids', 'style', 'speed' ], 
        output_names = [ 'audio' ],
        opset_version = 17, 
        dynamic_axes = {
            'input_ids': { 1: 'input_ids_len' }, 
            'audio': { 1: 'num_samples' }, 
        }, 
        do_constant_folding = True, 
    )

    print('export kokoro.onnx ok!')

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')

def load_input_ids(pipeline, text):
    if pipeline.lang_code in 'ab':
        _, tokens = pipeline.g2p(text)
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            if not ps:
                continue
    else:
        ps = pipeline.g2p(text)

    if len(ps) > 510:
        ps = ps[:510]

    input_ids = list(filter(lambda i: i is not None, map(lambda p: pipeline.model.vocab.get(p), ps)))
    print(f"text: {text} -> phonemes: {ps} -> input_ids: {input_ids}")
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return ps, input_ids

def load_voice(pipeline, voice, phonemes):
    pack = pipeline.load_voice(voice).to(model.device)
    return pack[len(phonemes) - 1]

def load_sample(model):
    pipeline = KPipeline(lang_code='a', model=model.model, device='cpu')
    text = '''
    In today's fast-paced tech world, building software applications has never been easier — thanks to AI-powered coding assistants.'
    '''
    voice = 'checkpoints/voices/af_heart.pt'

#    pipeline = KPipeline(lang_code='z', model=model.model, device='cpu')
#    text = '''
#    2月15日晚，猫眼专业版数据显示，截至发稿，《哪吒之魔童闹海》（或称《哪吒2》）今日票房已达7.8亿元，累计票房（含预售）超过114亿元。
#    '''
#    voice = 'checkpoints/voices/zf_xiaoxiao.pt'

    phonemes, input_ids = load_input_ids(pipeline, text)
    style = load_voice(pipeline, voice, phonemes)
    speed = torch.IntTensor([1])

    return input_ids, style, speed

from scipy.signal import get_window
import numpy as np
def post_process(audio):
    n_fft = 100
    window = torch.from_numpy(get_window('hann', n_fft, fftbins=True).astype(np.float32))

    forward_transform = torch.stft(audio, n_fft, window=window, return_complex=True)
    magnitude = torch.abs(forward_transform)
    phase = torch.angle(forward_transform)

    magnitude *= 3
    magnitude[:, 18:, :] = 0.0

    inverse_transform = torch.istft(magnitude * torch.exp(phase * 1j), n_fft, window=window)
    return inverse_transform

def inference_onnx(model, output):
    onnx_file = output + "/" + "kokoro.onnx"
    session = ort.InferenceSession(onnx_file)

    input_ids, style, speed = load_sample(model)

    outputs = session.run(None, {
        'input_ids': input_ids.numpy(), 
        'style': style.numpy(), 
        'speed': speed.numpy(), 
    })

    output = torch.from_numpy(outputs[0])
    print(f'output: {output.shape}')
    print(output)

    output = post_process(output)

    audio = output[0].numpy()
    sd.play(audio, 24000)
    sd.wait()

def check_model(model):
    input_ids, style, speed = load_sample(model)
    output = model(input_ids, style, speed)

    print(f'output: {output.shape}')
    print(output)

    output = post_process(output)

    audio = output[0].numpy()
    sd.play(audio, 24000)
    sd.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export kokoro Model to ONNX", add_help=True)
    parser.add_argument("--inference", "-t", help="test kokoro.onnx model", action="store_true")
    parser.add_argument("--check", "-m", help="check kokoro model", action="store_true")
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    model = KModel(config_file, checkpoint_path)
    model4onnx = Model(model).eval()

    if args.inference:
        inference_onnx(model4onnx, output_dir)
    elif args.check:
        check_model(model4onnx)
    else:
        export_onnx(model4onnx, output_dir)
