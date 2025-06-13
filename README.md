# FRENEMY

F.renemy R.econnaissance E.xplorer N.avigating E.nvironments, M.alfunctioning & Y.elling

An interactive voice-based assistant that guides users through bike assembly using AI-powered speech recognition and natural language processing.

## 🌟 Features

- **Voice Interaction**: Speak naturally with the assistant
- **Step-by-Step Guidance**: Clear instructions for bike assembly
- **AI-Powered Responses**: Intelligent and humorous responses
- **Speech Recognition**: Converts your voice to text
- **Natural Voice Output**: Speaks responses clearly

## 🛠️ Technologies Used

- **Speech-to-Text**: OpenAI's Whisper
- **Text-to-Speech**: CoquiTTS
- **Language Model**: Ollama with Llama3
- **Audio Processing**: SoundDevice
- **Conversation Management**: LangChain

## 📋 Assembly Steps

1. Unbox all parts
2. Attach front wheel
3. Install handlebars
4. Secure seat
5. Attach pedals
6. Check brakes
7. Pump tires
8. Test ride

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed and running
- Audio input/output devices

### Installation

#### Using uv (Recommended)

[uv][uv-link] is a fast Python package installer and resolver.

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/frenemy.git
   cd frenemy
   ```

3. Create and activate virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

#### Using pip (Traditional Method)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/frenemy.git
   cd frenemy
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Managing Dependencies

This project uses `pip-tools` for dependency management. Here's how to use it:

1. Install pip-tools:

   ```bash
   pip install pip-tools
   ```

2. Update requirements.txt:

   ```bash
   pip-compile --allow-unsafe requirements.in
   ```
   Note: The `--allow-unsafe` flag is required because some packages (like `setuptools`) are marked as unsafe but are necessary build dependencies.

3. Install dependencies:

   ```bash
   pip-sync requirements.txt
   ```

To add a new dependency:

1. Add it to `requirements.in`
2. Run `pip-compile --allow-unsafe requirements.in`
3. Run `pip-sync requirements.txt`

### Usage

1. Start the assistant:

   ```bash
   python main.py
   ```

2. Follow the voice prompts:
   - Speak clearly when prompted
   - Say "next" to move to the next step
   - Say "repeat" to hear the current step again
   - Say "skip" to skip the current step

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE][license-link] file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- CoquiTTS for text-to-speech
- Ollama for language model
- LangChain for conversation management

[uv-link]: https://github.com/astral-sh/uv
[license-link]: LICENSE