from .augmentations import (AudioAmplify, CenterCrop, ColorJitter,
                            EntityBoxCrop, EntityBoxFlip, EntityBoxRescale,
                            Flip, Fuse, Imgaug, MelSpectrogram, MultiGroupCrop,
                            MultiScaleCrop, Normalize, RandomCrop,
                            RandomRescale, RandomResizedCrop, RandomScale,
                            Resize, TenCrop, ThreeCrop, 
                            TemporalLocalShuffle) # 💥 1. 在这里导入你的算子
from .compose import Compose
from .formating import (Collect, FormatAudioShape, FormatShape, ImageToTensor,
                        Rename, ToDataContainer, ToTensor, Transpose)
from .loading import (AudioDecode, AudioDecodeInit, AudioFeatureSelector,
                      BuildPseudoClip, DecordDecode, DecordInit,
                      DenseSampleFrames, FrameSelector,
                      GenerateLocalizationLabels, ImageDecode,
                      LoadAudioFeature, LoadHVULabel, LoadLocalizationFeature,
                      LoadProposals, OpenCVDecode, OpenCVInit, PyAVDecode,
                      PyAVDecodeMotionVector, PyAVInit, RawFrameDecode,
                      SampleAVAFrames, SampleFrames, SampleProposalFrames,
                      UntrimmedSampleFrames, SnippetSampleFrames)

__all__ = [
    'SampleFrames', 'PyAVDecode', 'DecordDecode', 'DenseSampleFrames',
    'OpenCVDecode', 'FrameSelector', 'MultiGroupCrop', 'MultiScaleCrop',
    'RandomResizedCrop', 'RandomCrop', 'Resize', 'Flip', 'Fuse', 'Normalize',
    'ThreeCrop', 'CenterCrop', 'TenCrop', 'ImageToTensor', 'Transpose',
    'Collect', 'FormatShape', 'Compose', 'ToTensor', 'ToDataContainer',
    'GenerateLocalizationLabels', 'LoadLocalizationFeature', 'LoadProposals',
    'DecordInit', 'OpenCVInit', 'PyAVInit', 'SampleProposalFrames',
    'UntrimmedSampleFrames', 'RawFrameDecode', 'DecordInit', 'OpenCVInit',
    'PyAVInit', 'SampleProposalFrames', 'ColorJitter', 'LoadHVULabel',
    'SampleAVAFrames', 'AudioAmplify', 'MelSpectrogram', 'AudioDecode',
    'FormatAudioShape', 'LoadAudioFeature', 'AudioFeatureSelector',
    'AudioDecodeInit', 'EntityBoxFlip', 'EntityBoxCrop', 'EntityBoxRescale',
    'RandomScale', 'ImageDecode', 'BuildPseudoClip', 'RandomRescale',
    'PyAVDecodeMotionVector', 'Rename', 'Imgaug', 'SnippetSampleFrames',
    'TemporalLocalShuffle' # 💥 2. 在这里把它暴露给系统注册表
]