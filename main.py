'''
As a fully functional initial prototype, this script can be run as is alongside NVIDIA Broadcast. Simply install and startup NVIDIA broadcast, configure it to use the default denoising and echo reduction features, and keep the default audio input setting set below. https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/

This default POC mode can be replaced by imlementing additional proccess_fnc functions to do audio processing in line.

Future work is to include methods that are able to run directly on a mobile device (such as some traditional ML methods or simpler TensorFlow models) or rewriting this stack to directly leverage the NVIDIA Maxine functionality that powers NVIDIA Broadcast. https://developer.nvidia.com/maxine

Example code ported from the following sources:
    https://makersportal.com/blog/2018/8/23/recording-audio-on-the-raspberry-pi-with-python-and-a-usb-microphone
    https://people.csail.mit.edu/hubert/pyaudio/docs/
'''
import pyaudio
import wave

# The device name for the microphone, get by running in "GET_DEVICE_MODE"
input_device = 'Microphone (NVIDIA Broadcast)'
input_device_idx = -1 # Do not change, dynamically set based on name

# The device name for the speakers, get by running in "GET_DEVICE_MODE"
output_device = 'Headset (HX 831s)'
output_device_idx = -1 # Do not change, dynamically set based on name

def get_device_names():
    '''Human readable output of all input/output audio devices to Identify the correct device'''
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i).get('name'))

def get_device_idx(device_name):
    '''Return an audio device index, given a device name'''
    p = pyaudio.PyAudio()
    print(f'Getting device index for {device_name}')
    for i in range(p.get_device_count()):
        if device_name == p.get_device_info_by_index(i).get('name'):
            return p.get_device_info_by_index(i).get('index')

def get_io_devices(input_device, output_device):
    input_idx = get_device_idx(input_device)
    output_idx = get_device_idx(output_device)
    print(f'Determined device idx: {input_device}:{input_idx}, {output_device}:{output_idx}')
    return input_idx, output_idx

##################################################################
################### Start of Sample / Test Code ####################
##################################################################

def record_clip(input_device_idx, record_secs=10, audio_output_name='audio_sample.wav', sample_rate=44100, chunk=4096, chans=1, format=pyaudio.paInt16):
    '''Record <length> seconds worth of sample audio from <input_device>'''
    print(f'Recording {record_secs} seconds of sample audio from input device: {input_device_idx}')
    audio = pyaudio.PyAudio()

    # Create pyaudio stream
    stream = audio.open(format=format, rate=sample_rate, channels=chans, \
                        input_device_index=input_device_idx, input=True, \
                        frames_per_buffer=chunk)
    frames = []

    # loop through stream and append audio chunks to frame array
    for i in range(0, int((sample_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print(f'Finished recording sample audio, writing to {audio_output_name}')
    wavefile = wave.open(audio_output_name,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(format))
    wavefile.setframerate(sample_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    return frames

def play_sample_audio(output_device_idx, audio_file_name='audio_sample.wav', chunk=4096):
    '''Play an example audio file'''
    p = pyaudio.PyAudio()

    print(f'Playing audio file smaple to devices {audio_file_name}')
    wf = wave.open(audio_file_name, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

def play_sample_framebuffer(output_device_idx, audio_stream, sample_rate=44100, chunk=4096, chans=1, format=pyaudio.paInt16):
    '''Play an example audio data buffer'''
    p = pyaudio.PyAudio()

    print(f'Playing audio stream to sample devices')
    stream = p.open(format=format,
                    channels=chans,
                    rate=sample_rate,
                    output=True)

    for data in audio_stream:
        stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

def sample_stream(input_device_idx, output_device_idx, record_secs=10, audio_output_name='audio_sample.wav', sample_rate=44100, chunk=4096, chans=1, format=pyaudio.paInt16):
    '''For several seconds stream input from an audio device / microphone to an outputdevice / speaker'''
    print(f'Recording {record_secs} seconds of sample audio from input device: {input_device_idx} and playing back to device {output_device_idx}')
    audio = pyaudio.PyAudio()

    # Create pyaudio stream
    input_stream = audio.open(format=format, rate=sample_rate, channels=chans, \
                        input_device_index=input_device_idx, input=True, \
                        frames_per_buffer=chunk)
    output_stream = audio.open(format=format, rate=sample_rate, channels=chans, \
                        input_device_index=output_device_idx, output=True, \
                        frames_per_buffer=chunk)

    # loop through stream and append audio chunks to frame array
    for i in range(0, int((sample_rate/chunk)*record_secs)):
        output_stream.write(input_stream.read(chunk))
    input_stream.stop_stream()
    output_stream.stop_stream()
    input_stream.close()
    output_stream.close()
    audio.terminate()

    print(f'Finished recording/playing back sample audio')
##################################################################
################### End of Sample / Test Code ####################
##################################################################

def stream_audio(input_device_idx, output_device_idx, process_fnc, audio_output_name='audio_sample.wav', sample_rate=44100, chunk=4096, chans=1, format=pyaudio.paInt16):
    '''Continuosly take audio input, prosess it through <process_fnc>, and play it to an output device'''
    print(f'Recording from input device: {input_device_idx} and playing back to device {output_device_idx}')
    audio = pyaudio.PyAudio()

    # Create pyaudio stream
    input_stream = audio.open(format=format, rate=sample_rate, channels=chans, \
                        input_device_index=input_device_idx, input=True, \
                        frames_per_buffer=chunk)
    output_stream = audio.open(format=format, rate=sample_rate, channels=chans, \
                        input_device_index=output_device_idx, output=True, \
                        frames_per_buffer=chunk)

    # loop through stream and append audio chunks to frame array
    while True:
        output_stream.write(
            process_fnc(input_stream.read(chunk))
        )

    input_stream.stop_stream()
    output_stream.stop_stream()
    input_stream.close()
    output_stream.close()
    audio.terminate()

def passthrough_audio(stream_chunk):
    '''A passthrough function that does not alter audio and instead assumes something else like NVIDIA Broadcast is doing the heavy lifting'''
    return stream_chunk


if __name__ == "__main__":
    mode = "run"

    if mode == "setup":
        get_device_names()
        quit()

    print("Initializing Audio Devices") 
    input_device_idx, output_device_idx = get_io_devices(input_device, output_device)

    if mode == "test_setup":
        #recording = record_clip(input_device_idx)
        play_sample_audio(output_device_idx)
        play_sample_framebuffer(output_device_idx, recording)
        sample_stream(input_device_idx, output_device_idx, 1000000)
        quit()

    if mode == "run":
        stream_audio(input_device_idx, output_device_idx, passthrough_audio)
