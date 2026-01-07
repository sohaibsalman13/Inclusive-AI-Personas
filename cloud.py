import modal

# 1. Define the Container (The computer in the cloud)
image = (
    modal.Image.debian_slim()
    .pip_install("vllm")
    .run_commands(
        # Pre-download the model so it starts instantly later
        "pip install huggingface_hub",
        "huggingface-cli download VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct"
    )
)

app = modal.App("munich-student-bot", image=image)


# 2. Define the GPU Class
@app.cls(gpu="A10G", container_idle_timeout=300)  # Keeps GPU warm for 5 mins
class Model:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        # Load the Full (Non-GGUF) Model for maximum intelligence
        self.llm = LLM(
            model="VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct",
            dtype="float16"  # Fast on A10G
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=500,
            stop=["<|eot_id|>"]
        )

    @modal.method()
    def generate_text(self, prompt: str):
        # vLLM expects a prompt, returns a list of outputs
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

