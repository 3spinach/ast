#!/usr/bin/env python3
"""
AST (Audio Spectrogram Transformer) Inference Script for ESC50
This script performs inference on audio samples using a trained AST model on ESC50 dataset.
"""

import os
import sys
import json
import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import ASTModel


class ESC50Inference:
    """Class for performing inference with AST model trained on ESC50."""
    
    # ESC50 class labels
    ESC50_LABELS = [
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", 
        "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets", 
        "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", 
        "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing", 
        "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", 
        "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing", 
        "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", 
        "clock_alarm", "clock_tick", "glass_breaking", "helicopter", "chainsaw", 
        "siren", "car_horn", "engine", "train", "church_bells", "airplane", 
        "fireworks", "hand_saw"
    ]
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model parameters for ESC50
        self.label_dim = 50
        self.input_tdim = 512  # ESC50 uses 5-second clips at 16kHz with 10ms hop
        self.input_fdim = 128
        self.imagenet_pretrain = False  # Set to False when loading a trained model
        self.audioset_pretrain = False
        self.model_size = 'base384'
        
        # Audio processing parameters
        self.target_length = 512  # 5.12 seconds
        self.sample_rate = 16000
        self.n_fft = 400
        self.win_length = 400
        self.hop_length = 160  # 10ms hop
        self.n_mels = 128
        self.fmin = 50
        self.fmax = 8000
        
        # Normalization parameters (ESC50 statistics)
        # You can compute these from your training data or use AudioSet stats as approximation
        self.norm_mean = -6.845978
        self.norm_std = 5.5654526
        
        # Load the model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load the trained AST model."""
        # Create model
        model = ASTModel(
            label_dim=self.label_dim,
            fstride=10,
            tstride=10,
            input_fdim=self.input_fdim,
            input_tdim=self.input_tdim,
            imagenet_pretrain=self.imagenet_pretrain,
            audioset_pretrain=self.audioset_pretrain,
            model_size=self.model_size
        )
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random initialization.")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _wav2fbank(self, audio_path):
        """
        Convert audio file to log mel spectrogram (fbank features).
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            fbank: Log mel spectrogram tensor
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Compute mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0
        )
        
        mel_spec = mel_spectrogram(waveform)
        
        # Convert to log scale (dB)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Transpose to have time as first dimension
        fbank = log_mel_spec.squeeze(0).transpose(0, 1)
        
        # Pad or crop to target length
        n_frames = fbank.shape[0]
        if n_frames > self.target_length:
            # Crop from the middle
            start = (n_frames - self.target_length) // 2
            fbank = fbank[start:start + self.target_length, :]
        elif n_frames < self.target_length:
            # Pad with zeros
            pad_amount = self.target_length - n_frames
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad_amount))
        
        return fbank
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file for inference.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio tensor ready for model input
        """
        # Get fbank features
        fbank = self._wav2fbank(audio_path)
        
        # Normalize
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        
        # Add batch dimension
        fbank = fbank.unsqueeze(0)
        
        return fbank
    
    def predict(self, audio_path, top_k=5):
        """
        Perform inference on an audio file.
        
        Args:
            audio_path: Path to the audio file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess audio
        input_tensor = self.preprocess_audio(audio_path)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Apply sigmoid for multi-label classification
            # For single-label classification (ESC50), we can use softmax
            probabilities = torch.nn.functional.softmax(output, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=min(top_k, self.label_dim))
            
        # Convert to numpy
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Create results dictionary
        results = {
            'audio_file': audio_path,
            'predictions': []
        }
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            results['predictions'].append({
                'rank': i + 1,
                'class': self.ESC50_LABELS[idx],
                'class_id': int(idx),
                'probability': float(prob),
                'confidence': f"{float(prob) * 100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, audio_paths, top_k=5):
        """
        Perform inference on multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            top_k: Number of top predictions to return for each file
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, top_k=top_k)
                results.append(result)
                print(f"Processed: {audio_path}")
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                results.append({
                    'audio_file': audio_path,
                    'error': str(e)
                })
        
        return results


def main():
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description='AST ESC50 Inference Script')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to audio file or directory of audio files')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show (default: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference (default: cuda)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save predictions as JSON (optional)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = ESC50Inference(
        model_path=args.model_path,
        device=args.device
    )
    
    # Prepare list of audio files
    audio_files = []
    if os.path.isdir(args.audio_path):
        # Process all audio files in directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        for ext in audio_extensions:
            audio_files.extend(Path(args.audio_path).glob(f'*{ext}'))
        audio_files = [str(f) for f in audio_files]
    else:
        # Single audio file
        audio_files = [args.audio_path]
    
    if not audio_files:
        print(f"No audio files found in {args.audio_path}")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process")
    
    # Run inference
    if len(audio_files) == 1:
        # Single file inference
        results = inference_engine.predict(audio_files[0], top_k=args.top_k)
        
        # Print results
        print("\n" + "="*60)
        print(f"Predictions for: {results['audio_file']}")
        print("="*60)
        
        for pred in results['predictions']:
            print(f"Rank {pred['rank']}: {pred['class']:<20} "
                  f"(ID: {pred['class_id']:>2}) - {pred['confidence']:>7}")
    else:
        # Batch inference
        results = inference_engine.predict_batch(audio_files, top_k=args.top_k)
        
        # Print results for each file
        for result in results:
            if 'error' in result:
                print(f"\nError processing {result['audio_file']}: {result['error']}")
            else:
                print("\n" + "="*60)
                print(f"Predictions for: {result['audio_file']}")
                print("="*60)
                
                for pred in result['predictions']:
                    print(f"Rank {pred['rank']}: {pred['class']:<20} "
                          f"(ID: {pred['class_id']:>2}) - {pred['confidence']:>7}")
    
    # Save results to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results if len(audio_files) > 1 else [results], f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()