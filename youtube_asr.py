import argparse
from rich.progress import track
import whisper
import yt_dlp


def main(h):

    def download_audio(url, output_path=h.saved_audio):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': output_path
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])


    video_url = h.youtube_link

    # Download the audio
    download_audio(video_url)

    print("Audio downloaded successfully.")

    SAMPLE_RATE = 16000
    DURATION = 30
    CROP_LEN = SAMPLE_RATE * DURATION

    # load model
    model = whisper.load_model("medium", device="cuda")

    # load audio
    wav = whisper.load_audio("audio.wav")

    num_split = wav.shape[0] // CROP_LEN
    num_split = num_split if wav.shape[0] % CROP_LEN == 0 else num_split + 1

    text = ""

    for i in track(range(num_split)):

        # pad/trim it to fit 30 seconds
        wav_seg = whisper.pad_or_trim(wav[CROP_LEN * i:])

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(wav_seg).to(model.device)

        # decode the audio
        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(model, mel, options)
        text += result.text + str(" ")


    with open(h.saved_transcription, "w", encoding="utf-8") as txt:
        txt.write(text)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--youtube_link', default="https://www.youtube.com/watch?v=9w5kSufPTNI")
    parser.add_argument('--saved_audio', default="audio")
    parser.add_argument('--saved_transcription', default="transcription.txt")


    h = parser.parse_args()

    main(h)