from elevenlabs import  save, set_api_key

set_api_key("sk_4c17caa48cf53702ea9a291bf91a61c126df611f9d3fc116")

audio = generate(
    text="Hello world!",
    voice="Rachel",
    model="eleven_multilingual_v2"
)

save(audio, "output.wav")