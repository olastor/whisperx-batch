from whisperx.asr import load_model
from whisperx.alignment import align, load_align_model
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

class TranscriptionActor:
    def __init__(self, model_name, device, device_index, model_dir, compute_type, 
                 language, asr_options, vad_method, vad_options, task, 
                 model_cache_only, threads, batch_size, chunk_size, print_progress, verbose):
        self.model = load_model(
            model_name,
            device=device,
            device_index=device_index,
            download_root=model_dir,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
            vad_method=vad_method,
            vad_options=vad_options,
            task=task,
            local_files_only=model_cache_only,
            threads=threads,
        )
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.print_progress = print_progress
        self.verbose = verbose
    
    def __call__(self, item):
        audio = item["arr"]
        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            print_progress=self.print_progress,
            verbose=self.verbose,
        )
        item["transcription"] = result
        return item


class AlignmentActor:
    def __init__(self, align_language, device, align_model_name, interpolate_method, 
                 return_char_alignments, print_progress):
        self.align_model, self.align_metadata = load_align_model(
            align_language, device, model_name=align_model_name
        )
        self.device = device
        self.interpolate_method = interpolate_method
        self.return_char_alignments = return_char_alignments
        self.print_progress = print_progress
    
    def __call__(self, item):
        result = item["transcription"]
        audio = item["arr"]
        
        if self.align_model is not None and len(result["segments"]) > 0:
            result = align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                interpolate_method=self.interpolate_method,
                return_char_alignments=self.return_char_alignments,
                print_progress=self.print_progress,
            )
        
        item["transcription"] = result
        return item


class DiarizationActor:
    def __init__(self, diarize_model_name, hf_token, device, min_speakers, max_speakers, 
                 return_speaker_embeddings):
        self.diarize_model = DiarizationPipeline(
            model_name=diarize_model_name, 
            use_auth_token=hf_token, 
            device=device
        )
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.return_speaker_embeddings = return_speaker_embeddings
    
    def __call__(self, item):
        result = item["transcription"]
        audio = item["arr"]
        
        diarize_result = self.diarize_model(
            audio, 
            min_speakers=self.min_speakers, 
            max_speakers=self.max_speakers, 
            return_embeddings=self.return_speaker_embeddings
        )
        
        if self.return_speaker_embeddings:
            diarize_segments, speaker_embeddings = diarize_result
        else:
            diarize_segments = diarize_result
            speaker_embeddings = None
        
        result = assign_word_speakers(diarize_segments, result, speaker_embeddings)
        item["transcription"] = result
        return item

