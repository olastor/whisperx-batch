import argparse
import os
from os.path import abspath
import warnings
import numpy as np
import torch
import ray
from whisperx_batch.actors import AlignmentActor, DiarizationActor, TranscriptionActor
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer
from whisperx.log_utils import get_logger
from whisperx.audio import load_audio

logger = get_logger(__name__)


def load_audio_into_memory(item, device):
    audio_path = item["path"]
    audio = load_audio(audio_path)
    item["arr"] = audio
    return item


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    model_cache_only: bool = args.pop("model_cache_only")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")

    output_dir = abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = args.pop("task")
    if task == "translate":
        no_align = True

    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_method: str = args.pop("vad_method")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    diarize_model_name: str = args.pop("diarize_model")
    print_progress: bool = args.pop("print_progress")
    return_speaker_embeddings: bool = args.pop("speaker_embeddings")

    if return_speaker_embeddings and not diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{
                    args['language']}'; using English instead."
            )
        args["language"] = "en"
    align_language = (
        args["language"] if args["language"] is not None else "en"
    )

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "hotwords": args.pop("hotwords"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn(
            "--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    ray.init()

    transcription_config = {
        "model_name": model_name,
        "device": device,
        "device_index": device_index,
        "model_dir": model_dir,
        "compute_type": compute_type,
        "language": args["language"],
        "asr_options": asr_options,
        "vad_method": vad_method,
        "vad_options": {
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        "task": task,
        "model_cache_only": model_cache_only,
        "threads": faster_whisper_threads,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "print_progress": print_progress,
        "verbose": verbose,
    }

    alignment_config = None if no_align else {
        "align_language": align_language,
        "device": device,
        "align_model_name": align_model,
        "interpolate_method": interpolate_method,
        "return_char_alignments": return_char_alignments,
        "print_progress": print_progress,
    }

    diarization_config = None if not diarize else {
        "diarize_model_name": diarize_model_name,
        "hf_token": hf_token,
        "device": device,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "return_speaker_embeddings": return_speaker_embeddings,
    }

    audio_paths = args.pop("batch_file")

    with open(audio_paths, "r") as f:
        audio_paths = [abspath(line.strip())
                       for line in f.readlines() if line.strip()]

    raw_ds = ray.data.from_items([{"path": p} for p in audio_paths])
    ds = raw_ds.map(
        load_audio_into_memory,
        fn_kwargs={"device": device},
        num_cpus=args.pop("load_audio_num_cpus"),
        compute=ray.data.TaskPoolStrategy(
            size=args.pop("load_audio_num_tasks")
        )
    )
    ds = ds.map(
        TranscriptionActor,
        fn_constructor_kwargs=transcription_config,
        compute=ray.data.ActorPoolStrategy(
            size=args.pop("transcribe_actor_size")),
        num_gpus=args.pop("transcribe_num_gpus"),
        num_cpus=args.pop("transcribe_num_cpus")
    )

    if alignment_config:
        ds = ds.map(
            AlignmentActor,
            fn_constructor_kwargs=alignment_config,
            compute=ray.data.ActorPoolStrategy(
                size=args.pop("align_actor_size")),
            num_gpus=args.pop("align_num_gpus"),
            num_cpus=args.pop("align_num_cpus")
        )

    if diarization_config:
        ds = ds.map(
            DiarizationActor,
            fn_constructor_kwargs=diarization_config,
            compute=ray.data.ActorPoolStrategy(
                size=args.pop("diarize_actor_size")),
            num_gpus=args.pop("diarize_num_gpus"),
            num_cpus=args.pop("diarize_num_cpus")
        )

    for row in ds.iter_rows():
        del row["arr"]
        result = row["transcription"]
        result["language"] = align_language
        path = row["path"]
        writer(result, path, writer_args)

    ray.shutdown()
