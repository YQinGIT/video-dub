"""media_io — ffmpeg/ffprobe wrappers. PORTABLE (requires the ffmpeg binary).

Public API:
    probe(path)                       -> MediaInfo   inspect a media file
    extract_audio(video, out, ...)    -> Path        pull audio out of a video
    remux(video, audio, out)          -> Path        attach new audio to a video

All failures raise `videodub.errors.MediaIOError`.
"""

from videodub.media_io.convert import extract_audio, remux
from videodub.media_io.probe import MediaInfo, StreamInfo, probe

__all__ = ["MediaInfo", "StreamInfo", "extract_audio", "probe", "remux"]
