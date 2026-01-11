# Whisperx Batch

> [!WARNING]
> Consider this to be more of a proof-of-concept than a production-ready implementation!

---

This project leverages [Ray](https://www.ray.io/) for better batch processing of multiple audio files using [WhisperX](https://github.com/m-bain/whisperX).

**Main Features:**

- Avoid loading and resampling the same audio file multiple times
- Make use of pipelining to increase throughput and resource utilization
- Write output files as they become available dynamically and not just at the very end of execution
- Good observability and plenty of configuration options by delegating the scheduling to Ray.


The pipeline is divided into these stages:

- Load audio
- Transcribe ("TranscriptionActor")
- (optional) Align ("AlignmentActor")
- (optional) Diarize ("DiarizationActor")

You can see the current progress in the Dashboard ([http://localhost:8265/](http://localhost:8265/)).

Possible future improvements:

- Consider splitting stages further where it makes sense (requires some more complicated rewriting)
- Load audio as a tensor directly into the GPU in the beginnnig and reuse it in each stage instead of storing it the object that's passed through the pipeline


## Example Usage

**Notes:** 

- Use absolute paths for any argument that references a file or folder as it might be that the execution environment is changed by Ray.
- `--batch_file` and `--language` are required options
- Adjust resource params ("num_gpus", "num_cpus") and actor sizes for your environment. You might need to experiment a bit to achieve the best combination for optimal resource utilization and to prevent backpressure from happening.
- The batch file must contain one valid audio file path per line
- Original whisperX params should work the same, but it was not verified for all options. Diarization was not tested yet, for example. 

```bash
uv sync
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # see https://github.com/m-bain/whisperX/issues/1304

# Example execution on a single node with 4 CPU cores and a GPU with 16GB of vRAM (no diarization)
uv run whisperx-batch \
  --model large-v3 \
  --language en \
  --batch_size 16 \
  --output_dir /path/to/output \
  --batch_file /path/to/batchfile \
  --load_audio_num_tasks 2 \
  --load_audio_num_cpus 0.3 \
  --transcribe_actor_size 1 \
  --transcribe_num_gpus 0.4 \
  --transcribe_num_cpus 1 \
  --align_actor_size 2 \
  --align_num_gpus 0.3 \
  --align_num_cpus 1
```


## LICENSE

- The original whisperx CLI arguments as well as parts of the code in `batch.py` for initializing/configuring the models were copied from WhisperX and are licensed under [BSD 2-Clause License](https://github.com/m-bain/whisperX/blob/main/LICENSE)
- The new code added shall also be licensed under BSD 2-Clause License.
