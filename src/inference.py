import argparse
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


@torch.inference_mode()
def run_inference(image_path: str, model_dir: str):
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.eval()
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
        )

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # empty prompt (or set a task-specific prompt here)
    decoder_input_ids = processor.tokenizer(
        "", add_special_tokens=False, return_tensors="pt"
    ).input_ids

    gen = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=getattr(model.config, "max_length", 512),
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_beams=1,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    seq = processor.batch_decode(gen.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    return seq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir", required=True, help="trained model folder (results/run_xxx/model)"
    )
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(run_inference(args.image, args.model_dir))

