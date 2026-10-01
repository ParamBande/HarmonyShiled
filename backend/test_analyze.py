import urllib.request, json

# Test /analyze with an existing pilot AI file
import sys
sys.path.insert(0, '.')

url = 'http://localhost:8000/analyze'
filepath = r'D:\PROJECTS\harmonyshield\harmonyshield\data\pilot_dataset\ai\-0Gj8-vB1q4_MusicGen_medium.wav'

import urllib.parse, mimetypes

boundary = 'harmonyshield_test_boundary'
with open(filepath, 'rb') as f:
    audio_data = f.read()

body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="-0Gj8-vB1q4_MusicGen_medium.wav"\r\n'
    f'Content-Type: audio/wav\r\n\r\n'
).encode() + audio_data + f'\r\n--{boundary}--\r\n'.encode()

req = urllib.request.Request(
    url,
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    method='POST'
)

try:
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    print('=== ANALYZE RESULT ===')
    print(f'File: {result["file"]["name"]}')
    print(f'Duration: {result["file"]["duration"]:.2f}s')
    print(f'Classical: {result["classical"]["prediction"]} (AI={result["classical"]["ai_probability"]:.4f})')
    print(f'ResNet:    {result["resnet18"]["prediction"]} (AI={result["resnet18"]["ai_probability"]:.4f})')
    print(f'AST:       {result["ast"]["prediction"]} (AI={result["ast"]["ai_probability"]:.4f})')
    print(f'Consensus: {result["consensus"]["status"]}')
    print(f'Top features: {len(result["features"])}')
    print(f'Mel shape: {len(result["mel_spectrogram"])} rows x {len(result["mel_spectrogram"][0])} cols')
    print(f'Inference time: {result["inference_time_s"]}s')
    print('=== SUCCESS ===')
except Exception as e:
    print(f'ERROR: {e}')
